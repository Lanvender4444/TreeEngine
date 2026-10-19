"""TE-015..018 / TE-035: HTTP v1 contract, access rules, limits, errors, auth."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from treeengine import TreeEngine  # noqa: E402
from treeengine.api import schema  # noqa: E402
from treeengine.integrations.http import create_app  # noqa: E402
from treeengine.integrations.service import ServiceConfig  # noqa: E402

DOC = "# Ops\n\n## Cache\n\nRedis eviction uses allkeys-lru.\n\n## Deploy\n\nRoll out daily.\n"


def _validate(obj: Any, name: str) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(obj, schema(name))


@pytest.fixture
def env(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "allowed"
    root.mkdir()
    (root / "ops.md").write_text(DOC, encoding="utf-8")
    secret = tmp_path / "secret.md"
    secret.write_text("# Secret\n\npassword: hunter2\n", encoding="utf-8")
    eng = TreeEngine(tmp_path / "kb.db")
    yield {"root": root, "secret": secret, "eng": eng, "tmp": tmp_path}
    eng.close()


def _client(env: dict[str, Any], **kw: Any) -> TestClient:
    cfg = kw.pop("config", ServiceConfig(ingest_roots=[env["root"]]))
    return TestClient(create_app(env["eng"], cfg, **kw), raise_server_exceptions=False)


def _err(r: Any, status: int, code: str) -> None:
    assert r.status_code == status, (r.status_code, r.text)
    body = r.json()
    _validate(body, "error")
    assert body["error"]["code"] == code, body


def test_ingest_paths_are_confined(env: dict[str, Any]) -> None:
    root, secret = env["root"], env["secret"]
    locked = TestClient(create_app(env["eng"]), raise_server_exceptions=False)
    _err(locked.post("/v1/documents", json={"path": str(root / "ops.md")}), 403, "access_denied")
    c = _client(env)
    ok = c.post("/v1/documents", json={"path": str(root / "ops.md"), "metadata": {"team": "sre"}})
    assert ok.status_code == 200, ok.text
    doc = ok.json()
    _validate(doc, "document")
    assert doc["source_revision"].startswith("sha256:") and doc["metadata"]["team"] == "sre"
    _err(c.post("/v1/documents", json={"path": str(secret)}), 403, "access_denied")
    _err(c.post("/v1/documents", json={"path": str(root / ".." / "secret.md")}), 403,
         "access_denied")  # fmt: skip
    link = root / "link.md"
    try:
        os.symlink(secret, link)
    except OSError:
        pytest.skip("no symlinks here")
    _err(c.post("/v1/documents", json={"path": str(link)}), 403, "access_denied")
    _err(c.post("/v1/documents", json={"path": str(root / "missing.md")}), 404, "source_not_found")
    # the error never echoes the absolute path
    r = c.post("/v1/documents", json={"path": str(root / "missing.md")})
    assert str(env["tmp"]) not in r.text


def test_upload_and_limits(env: dict[str, Any]) -> None:
    c = _client(env, config=ServiceConfig(ingest_roots=[], max_upload_bytes=200))
    r = c.post("/v1/documents/upload?filename=notes.md", content=DOC.encode())
    assert r.status_code == 200, r.text
    assert r.json()["title"]
    _err(c.post("/v1/documents/upload?filename=big.md", content=b"x" * 500), 413, "resource_limit")
    _err(c.post("/v1/documents/upload?filename=a.exe", content=b"MZ"), 415, "unsupported_format")
    off = _client(env, config=ServiceConfig(allow_uploads=False))
    _err(off.post("/v1/documents/upload?filename=n.md", content=b"# x"), 403, "access_denied")


def test_retrieve_contract_and_degradation(env: dict[str, Any]) -> None:
    c = _client(env)
    doc = c.post("/v1/documents", json={"path": str(env["root"] / "ops.md")}).json()
    r = c.post("/v1/retrieve", json={"query": "Redis eviction", "context_tokens": 300})
    assert r.status_code == 200
    body = r.json()
    _validate(body, "retrieval_result")
    # balanced without an embedder: FTS only, and it says so
    assert body["status"] == "degraded" and body["retrieval"]["preset"] == "balanced"
    assert any("embedder" in d for d in body["retrieval"]["degradations"])
    span = body["context_spans"][0]
    assert (
        "allkeys-lru" in span["text"]
        and span["citation"]["source_revision"] == doc["source_revision"]
    )
    lite = c.post("/v1/retrieve", json={"query": "Redis", "preset": "lite"}).json()
    assert lite["status"] == "ok" and lite["retrieval"]["preset"] == "lite"
    none = c.post("/v1/retrieve", json={"query": "zzzzqqq", "preset": "lite"}).json()
    assert none["status"] == "no_results"
    _err(c.post("/v1/retrieve", json={"query": "x", "preset": "quality", "strict": True}), 501,
         "dependency_missing")  # fmt: skip
    _err(c.post("/v1/retrieve", json={"query": "x", "preset": "nope"}), 422, "invalid_request")
    _err(c.post("/v1/retrieve", json={"query": "x", "policy": {"nope": 1}}), 422,
         "invalid_request")  # fmt: skip
    _err(c.post("/v1/retrieve", json={"query": "x" * 5000}), 413, "resource_limit")
    _err(c.post("/v1/retrieve", json={"query": "x", "context_tokens": 10**6}), 413,
         "resource_limit")  # fmt: skip
    _err(c.post("/v1/retrieve", json={"query": "x", "document_id": "missing"}), 404,
         "document_not_found")  # fmt: skip
    _err(c.post("/v1/search", json={"query": "x", "limit": 10**4}), 413, "resource_limit")
    s = c.post("/v1/search", json={"query": "Redis", "limit": 2}).json()
    assert s["status"] == "ok" and s["evidence"][0]["source_revision"]
    for e in s["evidence"]:
        _validate(e, "evidence")


def test_navigation_and_context(env: dict[str, Any]) -> None:
    c = _client(env)
    doc = c.post("/v1/documents", json={"path": str(env["root"] / "ops.md")}).json()
    d = c.get(f"/v1/documents/{doc['id']}").json()
    nid = d["roots"][0]["id"]
    for n in d["roots"]:
        _validate(n, "node")
    kids = c.get(f"/v1/nodes/{nid}/children").json()
    assert kids
    page = c.get(f"/v1/nodes/{kids[0]['id']}/blocks?limit=1").json()
    assert page["blocks"] and "next_offset" in page
    _validate(page["blocks"][0], "block")
    _err(c.get("/v1/nodes/missing/children"), 404, "node_not_found")
    _err(c.get(f"/v1/nodes/{nid}/blocks?limit=100000"), 413, "resource_limit")
    ctx = c.post(
        "/v1/context",
        json={"hits": [{"document_id": doc["id"], "text": "Redis eviction uses"}]},
    ).json()
    assert ctx["context_spans"][0]["mapping"]["status"] == "exact"
    stale = c.post(
        "/v1/context",
        json={
            "hits": [
                {
                    "document_id": doc["id"],
                    "text": "Redis eviction uses",
                    "source_revision": "sha256:old",
                }
            ],  # fmt: skip
            "strict_mapping": True,
        },
    ).json()
    assert stale["status"] == "no_results"  # strict: a stale hit is dropped
    assert c.delete(f"/v1/documents/{doc['id']}").json()["deleted"] == doc["id"]
    _err(c.get(f"/v1/documents/{doc['id']}"), 404, "document_not_found")


def test_auth_and_errors(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client(env, auth_token="s3cret")
    assert c.get("/v1/health").status_code == 200  # liveness stays public
    assert c.get("/v1/documents").status_code == 401
    assert c.get("/v1/documents", headers={"Authorization": "Bearer nope"}).status_code == 401
    ok = c.get("/v1/documents", headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200
    assert c.post("/search", json={"query": "x"}).status_code == 401  # legacy routes too

    def boom(*a: Any, **k: Any) -> None:
        raise RuntimeError("disk at /very/secret/path exploded")

    monkeypatch.setattr(env["eng"], "search", boom)
    r = c.post("/v1/search", json={"query": "x"}, headers={"Authorization": "Bearer s3cret"})
    _err(r, 500, "treeengine_error")
    assert "secret" not in r.text


def test_openapi(env: dict[str, Any]) -> None:
    spec = _client(env).get("/openapi.json").json()
    assert {"/v1/retrieve", "/v1/context", "/v1/documents/upload"} <= set(spec["paths"])
    assert "/retrieve" not in spec["paths"]  # legacy routes are not part of the v1 contract
    import treeengine

    assert spec["info"]["version"] == treeengine.__version__


def test_server_refuses_public_bind_without_token() -> None:
    from treeengine.integrations.http import _loopback, main

    assert _loopback("127.0.0.1") and _loopback("::1") and _loopback("localhost")
    assert not _loopback("0.0.0.0")
    assert main(["--host", "0.0.0.0", "--db", ":memory:"]) == 2
