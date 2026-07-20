"""P13 portfolio, architecture, and CI structural tests (Epic E9, D-032-D-034).

These tests verify the presentation layer without asserting whole paragraphs:
that the architecture/portfolio files exist, their local links resolve, the
asset manifest matches the real PNGs and marks them synthetic, no real BRFSS
column or local path leaks into text assets, the CI workflow keeps its
least-privilege/no-secret/no-training contract, no premature badge exists, and
no unqualified overclaim appears. Both official artifact hashes documented in
the architecture page must match the committed artifacts.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from pathlib import Path

import pytest

from src.data import FEATURE_COLUMNS, PROJECT_ROOT, TARGET

DOCS = PROJECT_ROOT / "docs"
PORTFOLIO = DOCS / "p13-portfolio"
MANIFEST = PORTFOLIO / "assets_manifest.json"
ARCHITECTURE = DOCS / "architecture.md"
PORTFOLIO_SUMMARY = DOCS / "portfolio-summary.md"
README = PROJECT_ROOT / "README.md"
CI_WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
GIT_ATTRIBUTES = PROJECT_ROOT / ".gitattributes"

MODEL_ARTIFACT = PROJECT_ROOT / "models" / "diabetes_risk_model.joblib"
BACKGROUND_ASSET = PROJECT_ROOT / "models" / "shap_background_v1.json"

TEXT_ASSETS = (ARCHITECTURE, PORTFOLIO_SUMMARY, PORTFOLIO / "report.md", MANIFEST)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def png_dimensions(path: Path) -> tuple[int, int]:
    head = path.read_bytes()[:26]
    width, height = struct.unpack(">II", head[16:24])
    return int(width), int(height)


# ---------------------------------------------------------------------------
# Structure and links
# ---------------------------------------------------------------------------


def test_p13_documentation_files_exist():
    for path in (
        ARCHITECTURE,
        PORTFOLIO_SUMMARY,
        PORTFOLIO / "report.md",
        MANIFEST,
        CI_WORKFLOW,
    ):
        assert path.is_file(), f"missing P13 file: {path}"


@pytest.mark.parametrize("doc", [ARCHITECTURE, PORTFOLIO_SUMMARY, README])
def test_local_markdown_links_resolve(doc):
    text = read(doc)
    # [label](target) markdown links; keep only local (non-http, non-anchor).
    for target in re.findall(r"\]\(([^)]+)\)", text):
        target = target.split("#", 1)[0].strip()
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        resolved = (doc.parent / target).resolve()
        assert resolved.exists(), f"broken link in {doc.name}: {target}"


def test_architecture_documents_both_official_hashes():
    text = read(ARCHITECTURE)
    assert sha256(MODEL_ARTIFACT) in text, "model SHA-256 missing/incorrect"
    assert sha256(BACKGROUND_ASSET) in text, "background SHA-256 missing/incorrect"
    # The background is a hash-bound text artifact whose reviewed runtime bytes
    # use CRLF.  Materialize those same bytes on Windows and Linux checkouts.
    assert (
        "/models/shap_background_v1.json text eol=crlf"
        in read(GIT_ATTRIBUTES)
    )
    normalized = " ".join(text.split())
    assert "not a hard-coded allowlist or authenticity check" in normalized
    assert "cannot make an untrusted pickle" in normalized


# ---------------------------------------------------------------------------
# Asset manifest and privacy
# ---------------------------------------------------------------------------


def test_manifest_matches_pngs_and_marks_them_synthetic():
    manifest = json.loads(read(MANIFEST))
    assert manifest["decision"] == "D-033"
    entries = manifest["assets"]
    assert entries, "manifest has no assets"

    manifest_names = {e["name"] for e in entries}
    png_names = {p.name for p in PORTFOLIO.glob("*.png")}
    assert manifest_names == png_names, (
        f"manifest/PNG mismatch: only-manifest={manifest_names - png_names}, "
        f"only-png={png_names - manifest_names}"
    )

    for entry in entries:
        png = PORTFOLIO / entry["name"]
        assert png.is_file()
        assert entry["contains_real_rows"] is False
        assert entry["sha256"] == sha256(png), f"stale sha256 for {entry['name']}"
        width, height = png_dimensions(png)
        assert entry["dimensions"] == {"width": width, "height": height}
        for key in ("purpose", "viewport", "synthetic_source", "alt_text", "date"):
            assert entry.get(key), f"{entry['name']} missing {key}"
        # Alt text must be a useful sentence, not a stub.
        assert len(entry["alt_text"]) >= 40


def test_no_local_paths_in_text_assets():
    # Raw home paths and usernames must never leak into committed portfolio text.
    for path in (*TEXT_ASSETS, README):
        lowered = read(path).lower()
        for forbidden in ("c:\\users", "c:/users", "/home/", "/users/diego",
                          "arcecastillogpt"):
            assert forbidden not in lowered, f"{path.name} leaks a local path"


def test_portfolio_assets_do_not_dump_the_target_or_real_rows():
    # The text/manifest and allowed file set are structural privacy guards. The
    # screenshots intentionally show declared synthetic probabilities, so
    # pixel-level provenance remains a documented capture + visual-review gate.
    # (The architecture page may legitimately name the target as schema.)
    for path in (MANIFEST, PORTFOLIO / "report.md"):
        assert TARGET not in read(path), f"{path.name} references the target column"
    privacy = json.loads(read(MANIFEST))["privacy"].lower()
    assert "synthetic demonstration probabilities are intentionally visible" in privacy
    assert "probability associated with a real/user row" in privacy


def test_portfolio_directory_has_no_committed_data_files():
    # Only PNGs, the manifest, and the report belong here -- no CSV/parquet/joblib
    # (a real or synthetic data dump would be out of place and risky).
    for path in PORTFOLIO.iterdir():
        assert path.suffix.lower() not in (".csv", ".parquet", ".joblib", ".pkl"), (
            f"unexpected data file in portfolio: {path.name}"
        )


# ---------------------------------------------------------------------------
# Claims discipline (allow the negated form, forbid the affirmative claim)
# ---------------------------------------------------------------------------

_SENSITIVE_PHRASES = (
    "clinically validated",
    "fair model",
    "unbiased",
    "production-grade",
    "secure artifact loading",
)
_NEGATIONS = ("not", "never", "no ", "n't", "without", "makes no", "isn")


@pytest.mark.parametrize("doc", [ARCHITECTURE, PORTFOLIO_SUMMARY, README,
                                 PORTFOLIO / "report.md"])
def test_no_unqualified_overclaims(doc):
    lowered = read(doc).lower()
    for phrase in _SENSITIVE_PHRASES:
        start = 0
        while True:
            idx = lowered.find(phrase, start)
            if idx == -1:
                break
            window = lowered[max(0, idx - 24):idx]
            assert any(neg in window for neg in _NEGATIONS), (
                f"{doc.name} uses '{phrase}' without a qualifier"
            )
            start = idx + len(phrase)


# ---------------------------------------------------------------------------
# CI workflow contract (D-034)
# ---------------------------------------------------------------------------


def test_ci_workflow_is_least_privilege_and_pinned():
    text = read(CI_WORKFLOW)
    lowered = text.lower()
    assert "permissions:" in lowered and "contents: read" in lowered
    assert 'python-version: "3.12"' in text or "python-version: '3.12'" in text
    assert "pip install -r requirements.txt" in text
    assert "pip check" in text
    assert "pytest tests" in text
    assert "timeout-minutes:" in lowered
    # Triggers.
    for trigger in ("push", "pull_request", "workflow_dispatch"):
        assert trigger in lowered


def test_ci_workflow_does_no_training_secrets_or_artifact_writes():
    # Check executable content only: drop YAML comment lines (which may mention
    # these tokens in a "does not ..." explanation).
    lines = [
        line for line in read(CI_WORKFLOW).splitlines()
        if not line.lstrip().startswith("#")
    ]
    lowered = "\n".join(lines).lower()
    for forbidden in (
        "src.artifacts",       # artifact generation
        "src.explainability",  # SHAP/background regeneration
        "src.fairness",        # audit regeneration
        "secrets.",            # any repository secret
        "kaggle",              # dataset credentials/download
        "git commit",
        "git push",
        "train",
    ):
        assert forbidden not in lowered, f"CI must not reference '{forbidden}'"


def test_no_premature_ci_badge():
    # Per D-034, no status badge until a real remote green run exists.
    readme = read(README).lower()
    for badge_marker in ("workflows/ci.yml/badge.svg", "img.shields.io",
                        "actions/workflows/ci.yml/badge"):
        assert badge_marker not in readme, "CI badge must not appear before a green run"


# ---------------------------------------------------------------------------
# Contract preservation surfaced in the portfolio text
# ---------------------------------------------------------------------------


def test_reference_displays_are_stated_and_unchanged():
    for path in (PORTFOLIO_SUMMARY, README):
        text = read(path)
        for display in ("0.3%", "60.0%", "70.0%", "79.9%"):
            assert display in text, f"{path.name} missing reference display {display}"


def test_feature_contract_is_referenced_not_redefined():
    # The architecture page should describe the 21-feature contract by count,
    # not silently redefine it.
    assert "21" in read(ARCHITECTURE)
    assert len(FEATURE_COLUMNS) == 21
