"""Command-line interface.

pt2en translate curso.pdf -o course.pdf --style technical
pt2en serve --port 8000
pt2en sample exemplo.pdf
pt2en doctor
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from pt2en import __version__
from pt2en.config import (
    EnglishVariant,
    QAReviewMode,
    Settings,
    TranslationStyle,
    env_files_found,
    get_settings,
    settings_warning,
)
from pt2en.errors import PipelineError


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


def cmd_translate(args: argparse.Namespace, settings: Settings) -> int:
    from pt2en.pipeline.options import JobOptions
    from pt2en.pipeline.orchestrator import translate_file

    src = Path(args.input)
    if not src.exists():
        print(f"error: {src} does not exist", file=sys.stderr)
        return 2
    out = Path(args.output) if args.output else src.with_name(src.stem + "-en.pdf")
    glossary = Path(args.glossary).read_text(encoding="utf-8") if args.glossary else ""
    options = JobOptions.from_settings(
        settings,
        provider=args.provider,
        style=TranslationStyle(args.style) if args.style else None,
        english_variant=EnglishVariant(args.variant) if args.variant else None,
        preserve_terminology=False if args.no_preserve_terms else None,
        localize_numbers=True if args.localize_numbers else None,
        translate_images=False if args.no_images else None,
        ocr_enabled=False if args.no_ocr else None,
        qa_review=QAReviewMode(args.qa) if args.qa else None,
        glossary_text=glossary,
    )
    last = {"stage": None}

    def progress(ev) -> None:
        if args.quiet:
            return
        if ev.stage != last["stage"]:
            last["stage"] = ev.stage
            print(f"\n[{ev.stage_label}]", file=sys.stderr)
        eta = f" · ~{ev.eta_seconds:.0f}s left" if ev.eta_seconds else ""
        print(
            f"\r  {ev.overall * 100:5.1f}%  {ev.message[:70]:<70}{eta}",
            end="",
            file=sys.stderr,
            flush=True,
        )

    t0 = time.time()
    try:
        result = translate_file(src.read_bytes(), settings, options, progress=progress)
    except PipelineError as exc:
        print(f"\nerror: {exc.message}", file=sys.stderr)
        if exc.detail:
            print(f"detail: {exc.detail}", file=sys.stderr)
        return 1
    out.write_bytes(result.output_pdf)
    report_path = Path(args.report) if args.report else out.with_suffix(".qa.json")
    report_path.write_text(
        json.dumps(result.report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.review:
        Path(args.review).write_bytes(result.review_pdf)
    s = result.report["summary"]
    print(
        f"\n\nTranslated PDF : {out}\nQA report      : {report_path}"
        + (f"\nReview PDF     : {args.review}" if args.review else "")
        + f"\nStatus         : {s['status']} (score {s['score']}/100; {s['errors']} errors, "
        f"{s['warnings']} warnings)\nTime           : {time.time() - t0:.1f}s",
        file=sys.stderr,
    )
    for issue in result.report["issues"][:15]:
        page = f"p.{issue['page_number']}" if issue.get("page_number") else "doc"
        print(
            f"  - [{issue['severity']}] {page}: {issue['message'][:120]}",
            file=sys.stderr,
        )
    return 0


def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    uvicorn.run(
        "pt2en.web.app:create_app",
        factory=True,
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
    )
    return 0


def cmd_sample(args: argparse.Namespace, settings: Settings) -> int:
    from pt2en.samples import make_sample_pdf

    make_sample_pdf(args.output)
    print(f"Sample course PDF written to {args.output}")
    return 0


def cmd_doctor(args: argparse.Namespace, settings: Settings) -> int:
    from pt2en.ocr.tesseract import create_ocr
    from pt2en.reconstruction.fonts import FAMILY_FILES, FontResolver
    from pt2en.translation.registry import list_providers

    ok = True
    print(f"pt2en-translator {__version__}")
    ocr = create_ocr(
        settings.ocr_engine, settings.ocr_languages, settings.tesseract_cmd
    )
    if ocr.available():
        where = f" at {ocr.path}" if getattr(ocr, "path", None) else ""
        print(
            f"  [ok]   OCR: {ocr.name}{where} "
            f"(languages: {getattr(ocr, 'languages', '')})"
        )
        if "por" not in getattr(ocr, "languages", ""):
            print(f"  [warn] {ocr.reason or 'Portuguese OCR data missing'}")
    else:
        ok = False
        print(f"  [fail] OCR unavailable: {ocr.reason}")
    fonts = FontResolver(None, extra_dirs=settings.font_dirs)
    for family in FAMILY_FILES:
        found = fonts.family_file(family)
        status = found or "not found (Base-14 fallback used)"
        print(f"  [{'ok' if found else 'warn'}]   font family {family}: {status}")
    for p in list_providers(settings):
        mark = "ok" if p["available"] else "--"
        default = " (default)" if p["default"] else ""
        why = f" ({p['problem']})" if p["problem"] else ""
        print(f"  [{mark}]   provider {p['name']}{default}: {p['label']}{why}")
    files = env_files_found()
    for path in files:
        print(f"  [ok]   settings: {path}")
    warning = settings_warning(settings)
    if warning:
        ok = False
        print(f"  [fail] {warning}")
    print(f"  data dir: {settings.data_dir.resolve()}")
    return 0 if ok else 1


def cmd_models(args: argparse.Namespace, settings: Settings) -> int:
    """List (and optionally test) the chat models of the OpenAI-compatible endpoint."""
    from pt2en.errors import ProviderConfigurationError
    from pt2en.translation.base import TranslationError
    from pt2en.translation.providers.openai_provider import (
        OpenAICompatibleTranslator,
        probe_model,
        rank_chat_models,
    )

    try:
        provider = OpenAICompatibleTranslator(settings)
        ranked = rank_chat_models(provider.available_models())
    except (ProviderConfigurationError, TranslationError) as exc:
        print(f"error: {getattr(exc, 'message', exc)}")
        return 1
    where = "NVIDIA" if provider.nvidia else (settings.openai_endpoint or "OpenAI")
    print(f"{len(ranked)} chat models at {where}, best for translation first:")
    checked = 0
    for model in ranked:
        status = ""
        if args.check and checked < args.check:
            checked += 1
            status = "  " + probe_model(provider, model)
        print(f"  {model}{status}")
    print(
        "\nTo use one, set PT2EN_OPENAI_MODEL=<id> in .env and restart pt2en serve."
        "\nLeave it empty to let the app pick the first working model automatically."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pt2en", description="Layout-preserving PT-PT to English PDF translator"
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("translate", help="translate a PDF")
    t.add_argument("input")
    t.add_argument("-o", "--output")
    t.add_argument("--provider", help="anthropic | openai | deepl | argos | demo")
    t.add_argument("--style", choices=[s.value for s in TranslationStyle])
    t.add_argument("--variant", choices=[v.value for v in EnglishVariant])
    t.add_argument("--glossary", help="glossary file (pt = en per line, CSV or JSON)")
    t.add_argument("--no-preserve-terms", action="store_true")
    t.add_argument("--localize-numbers", action="store_true")
    t.add_argument(
        "--no-images", action="store_true", help="do not translate text inside images"
    )
    t.add_argument("--no-ocr", action="store_true")
    t.add_argument("--qa", choices=[m.value for m in QAReviewMode])
    t.add_argument("--report", help="QA report JSON path")
    t.add_argument("--review", help="annotated review PDF path")
    t.add_argument("-q", "--quiet", action="store_true")
    t.set_defaults(func=cmd_translate)

    s = sub.add_parser("serve", help="run the web application")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=cmd_serve)

    m = sub.add_parser("sample", help="generate the Portuguese sample course PDF")
    m.add_argument("output", nargs="?", default="curso_exemplo_pt.pdf")
    m.set_defaults(func=cmd_sample)

    d = sub.add_parser("doctor", help="check OCR, fonts and provider configuration")
    d.set_defaults(func=cmd_doctor)

    ml = sub.add_parser(
        "models", help="list the chat models your OpenAI-compatible/NVIDIA key can use"
    )
    ml.add_argument(
        "--check",
        type=int,
        nargs="?",
        const=10,
        default=0,
        metavar="N",
        help="send a one-word test request to the first N models (default 10)",
    )
    ml.set_defaults(func=cmd_models)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    _setup_logging(settings.log_level)
    return args.func(args, settings)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
