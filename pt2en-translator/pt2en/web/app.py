"""FastAPI application: REST API, server-sent progress events and the web UI."""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pymupdf
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from pt2en import __version__
from pt2en.config import (
    EnglishVariant,
    QAReviewMode,
    Settings,
    TranslationStyle,
    get_settings,
)
from pt2en.errors import InvalidDocumentError
from pt2en.pipeline.options import JobOptions
from pt2en.translation.glossary import parse_user_glossary
from pt2en.translation.registry import list_providers
from pt2en.web.jobs import TERMINAL, JobManager

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or get_settings()
    manager = JobManager(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async def janitor():
            while True:
                await asyncio.sleep(3600)
                manager.cleanup()

        task = asyncio.create_task(janitor())
        yield
        task.cancel()
        manager.shutdown()

    app = FastAPI(
        title="PT→EN Course PDF Translator", version=__version__, lifespan=lifespan
    )
    app.state.settings = settings
    app.state.jobs = manager

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                o.strip() for o in settings.cors_origins.split(",") if o.strip()
            ],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def require_token(
        request: Request, token: Optional[str] = Query(default=None)
    ) -> None:
        expected = settings.access_token
        if not expected:
            return
        header = request.headers.get("authorization", "")
        supplied = header[7:] if header.lower().startswith("bearer ") else (token or "")
        if not secrets.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail="Access token required.")

    def job_or_404(job_id: str):
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return job

    # ----------------------------------------------------------- pages
    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"}
        )

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # ------------------------------------------------------------- api
    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/api/config", dependencies=[Depends(require_token)])
    def config() -> dict:
        from pt2en.ocr.tesseract import create_ocr

        ocr = create_ocr(
            settings.ocr_engine, settings.ocr_languages, settings.tesseract_cmd
        )
        providers = list_providers(settings)
        default = next(
            (p["name"] for p in providers if p["default"]), settings.translator
        )
        if not settings.provider_available(default):
            default = next(
                (
                    p["name"]
                    for p in providers
                    if p["available"] and p["name"] != "demo"
                ),
                "demo",
            )
        return {
            "version": __version__,
            "providers": providers,
            "default_provider": default,
            "anthropic_model": settings.anthropic_model,
            "ocr": {
                "available": ocr.available(),
                "languages": getattr(ocr, "languages", ""),
            },
            "limits": {
                "max_upload_mb": settings.max_upload_mb,
                "max_pages": settings.max_pages,
            },
            "defaults": JobOptions.from_settings(settings, provider=default).model_dump(
                mode="json"
            ),
            "styles": [s.value for s in TranslationStyle],
            "variants": [v.value for v in EnglishVariant],
            "qa_modes": [m.value for m in QAReviewMode],
            "auth_required": bool(settings.access_token),
        }

    @app.post("/api/glossary/parse", dependencies=[Depends(require_token)])
    def glossary_parse(text: str = Form(default="")) -> dict:
        entries = parse_user_glossary(text)
        return {"entries": [e.to_dict() for e in entries], "count": len(entries)}

    @app.post("/api/jobs", status_code=201, dependencies=[Depends(require_token)])
    async def create_job(
        file: UploadFile = File(...), options: str = Form(default="{}")
    ) -> dict:
        limit = settings.max_upload_mb * 1024 * 1024
        chunks = []
        size = 0
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                raise HTTPException(
                    status_code=413,
                    detail=f"The file is larger than {settings.max_upload_mb} MB.",
                )
            chunks.append(chunk)
        try:
            raw = json.loads(options or "{}")
            if not isinstance(raw, dict):
                raise TypeError("options must be a JSON object")
            base = JobOptions.from_settings(settings).model_dump(mode="json")
            base.update(
                {
                    k: v
                    for k, v in raw.items()
                    if v is not None and k in JobOptions.model_fields
                }
            )
            opts = JobOptions.model_validate(base)
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            raise HTTPException(
                status_code=422, detail=f"Invalid options: {exc}"
            ) from exc
        provider = opts.resolved_provider(settings)
        if not settings.provider_available(provider):
            raise HTTPException(
                status_code=422,
                detail=f"The translation provider '{provider}' is not configured on the server.",
            )
        try:
            job = await asyncio.to_thread(
                manager.create, file.filename or "document.pdf", b"".join(chunks), opts
            )
        except InvalidDocumentError as exc:
            raise HTTPException(status_code=400, detail=exc.message) from exc
        return manager.public(job.id)

    @app.get("/api/jobs/{job_id}", dependencies=[Depends(require_token)])
    def get_job(job_id: str) -> dict:
        job_or_404(job_id)
        return manager.public(job_id)

    @app.get("/api/jobs/{job_id}/events", dependencies=[Depends(require_token)])
    async def job_events(job_id: str, request: Request) -> StreamingResponse:
        job_or_404(job_id)
        queue = manager.subscribe(job_id)

        async def stream():
            try:
                yield _sse({"type": "snapshot", "job": manager.public(job_id)})
                job = manager.get(job_id)
                if job is None or job.status in TERMINAL:
                    return
                while True:
                    if await request.is_disconnected():
                        return
                    try:
                        message = await asyncio.wait_for(queue.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    yield _sse(message)
                    if message.get("type") == "status" and message["job"]["status"] in {
                        s.value for s in TERMINAL
                    }:
                        return
            finally:
                manager.unsubscribe(job_id, queue)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/jobs/{job_id}/cancel", dependencies=[Depends(require_token)])
    def cancel_job(job_id: str) -> dict:
        job_or_404(job_id)
        return {"cancelled": manager.cancel(job_id)}

    @app.delete("/api/jobs/{job_id}", dependencies=[Depends(require_token)])
    def delete_job(job_id: str) -> dict:
        job_or_404(job_id)
        return {"deleted": manager.delete(job_id)}

    @app.get("/api/jobs/{job_id}/report", dependencies=[Depends(require_token)])
    def job_report(job_id: str) -> Response:
        job_or_404(job_id)
        path = manager.path(job_id, "report.json")
        if not path.exists():
            raise HTTPException(
                status_code=404, detail="The report is not available yet."
            )
        return Response(path.read_bytes(), media_type="application/json")

    DOWNLOADS = {
        "translated": ("output.pdf", "application/pdf", "{stem}-EN.pdf"),
        "review": ("review.pdf", "application/pdf", "{stem}-EN-review.pdf"),
        "original": ("input.pdf", "application/pdf", "{stem}.pdf"),
        "report": ("report.json", "application/json", "{stem}-EN-qa-report.json"),
    }

    @app.get(
        "/api/jobs/{job_id}/download/{kind}", dependencies=[Depends(require_token)]
    )
    def download(job_id: str, kind: str, inline: bool = False) -> FileResponse:
        job = job_or_404(job_id)
        if kind not in DOWNLOADS:
            raise HTTPException(status_code=404, detail="Unknown file.")
        name, media, pattern = DOWNLOADS[kind]
        path = manager.path(job_id, name)
        if not path.exists():
            raise HTTPException(
                status_code=404, detail="The file is not available yet."
            )
        filename = pattern.format(stem=Path(job.filename).stem)
        return FileResponse(
            path,
            media_type=media,
            filename=filename,
            content_disposition_type="inline" if inline else "attachment",
        )

    @app.get("/api/jobs/{job_id}/pages", dependencies=[Depends(require_token)])
    def page_sizes(job_id: str) -> dict:
        job_or_404(job_id)
        out = {}
        for doc_name, file in (("original", "input.pdf"), ("translated", "output.pdf")):
            path = manager.path(job_id, file)
            if path.exists():
                with pymupdf.open(path) as doc:
                    out[doc_name] = [
                        {"width": p.rect.width, "height": p.rect.height} for p in doc
                    ]
        return out

    @app.get(
        "/api/jobs/{job_id}/pages/{page}.png", dependencies=[Depends(require_token)]
    )
    async def page_image(
        job_id: str,
        page: int,
        doc: str = Query(default="translated", pattern="^(original|translated)$"),
        scale: float = Query(default=1.5, ge=0.5, le=3.0),
    ) -> Response:
        job_or_404(job_id)
        file = "input.pdf" if doc == "original" else "output.pdf"
        path = manager.path(job_id, file)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Document not available.")
        scale = round(scale * 4) / 4
        cache = manager.path(job_id, f"previews/{doc}-{page}-{scale}.png")
        if not cache.exists():

            def render() -> bytes:
                with pymupdf.open(path) as pdf:
                    if page < 1 or page > pdf.page_count:
                        raise HTTPException(
                            status_code=404, detail="Page out of range."
                        )
                    pix = pdf[page - 1].get_pixmap(
                        matrix=pymupdf.Matrix(scale, scale), alpha=False
                    )
                    return pix.tobytes("png")

            data = await asyncio.to_thread(render)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(data)
        return Response(
            cache.read_bytes(),
            media_type="image/png",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @app.get("/api/sample", dependencies=[Depends(require_token)])
    def sample() -> Response:
        from pt2en.samples import make_sample_pdf

        return Response(
            make_sample_pdf(),
            media_type="application/pdf",
            headers={
                "Content-Disposition": 'attachment; filename="curso_exemplo_pt.pdf"'
            },
        )

    @app.exception_handler(InvalidDocumentError)
    async def invalid_doc(_: Request, exc: InvalidDocumentError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": exc.message})

    return app


def _sse(message: dict) -> str:
    return f"data: {json.dumps(message, ensure_ascii=False)}\n\n"
