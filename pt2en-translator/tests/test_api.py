import json
import time

import pytest
from fastapi.testclient import TestClient

from pt2en.web.app import create_app


@pytest.fixture()
def client(settings):
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


def wait_for(client, job_id, timeout=240):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(0.5)
    raise AssertionError("job did not finish")


def test_config_and_health(client):
    assert client.get("/api/health").json()["status"] == "ok"
    cfg = client.get("/api/config").json()
    names = {p["name"]: p for p in cfg["providers"]}
    assert names["demo"]["available"]
    assert cfg["default_provider"] == "demo"
    assert set(cfg["styles"]) == {"academic", "technical", "literal"}
    assert client.get("/").status_code == 200


def test_rejects_non_pdf(client):
    r = client.post(
        "/api/jobs", files={"file": ("x.pdf", b"not a pdf", "application/pdf")}
    )
    assert r.status_code == 400
    assert "not a PDF" in r.json()["detail"]


def test_rejects_unconfigured_provider(client, sample_pdf):
    r = client.post(
        "/api/jobs",
        files={"file": ("curso.pdf", sample_pdf, "application/pdf")},
        data={"options": json.dumps({"provider": "anthropic"})},
    )
    assert r.status_code == 422


def test_full_job_lifecycle(client, sample_pdf):
    options = {
        "provider": "demo",
        "style": "technical",
        "glossary_text": "turma = class",
        "qa_review": "report",
    }
    r = client.post(
        "/api/jobs",
        files={"file": ("curso exemplo.pdf", sample_pdf, "application/pdf")},
        data={"options": json.dumps(options)},
    )
    assert r.status_code == 201, r.text
    job = r.json()
    assert job["page_count"] == 3 and job["options"]["style"] == "technical"
    done = wait_for(client, job["id"])
    assert done["status"] == "completed", done.get("error")
    assert done["files"]["output"] and done["summary"]["score"] >= 0

    pdf = client.get(f"/api/jobs/{job['id']}/download/translated")
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    assert "-EN.pdf" in pdf.headers["content-disposition"]
    report = client.get(f"/api/jobs/{job['id']}/report").json()
    assert any(g["pt"] == "turma" for g in report["glossary"])
    pages = client.get(f"/api/jobs/{job['id']}/pages").json()
    assert len(pages["original"]) == len(pages["translated"]) == 3
    png = client.get(f"/api/jobs/{job['id']}/pages/1.png?doc=translated&scale=1")
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
    assert client.get(f"/api/jobs/{job['id']}/pages/9.png").status_code == 404

    # Server-sent events end immediately for finished jobs with a snapshot.
    with client.stream("GET", f"/api/jobs/{job['id']}/events") as s:
        body = b"".join(s.iter_bytes())
    assert b'"snapshot"' in body

    assert client.delete(f"/api/jobs/{job['id']}").json()["deleted"]
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404


def test_access_token(settings, sample_pdf):
    settings.access_token = "secret"
    with TestClient(create_app(settings)) as c:
        assert c.get("/api/config").status_code == 401
        assert (
            c.get("/api/config", headers={"Authorization": "Bearer secret"}).status_code
            == 200
        )
        assert c.get("/api/config?token=secret").status_code == 200
