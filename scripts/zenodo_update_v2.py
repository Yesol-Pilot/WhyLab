# -*- coding: utf-8 -*-
"""Update existing Zenodo record #18948929 with new version."""
import requests
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load token
for env_path in [
    Path(r"d:\00.test\PAPER\WhyLab\.env"),
    Path(r"d:\00.test\PAPER\EthicaAI\.env"),
]:
    if env_path.exists():
        load_dotenv(env_path)
        if os.getenv("ZENODO_ACCESS_TOKEN"):
            break

TOKEN = os.getenv("ZENODO_ACCESS_TOKEN", "")
if not TOKEN:
    print("ERROR: No ZENODO_ACCESS_TOKEN found")
    sys.exit(1)

RECORD_ID = 18948929
API = "https://zenodo.org/api"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
PDF_PATH = Path(r"d:\00.test\PAPER\WhyLab\paper\main.pdf")

print(f"Token: ...{TOKEN[-4:]}")
print(f"Record: {RECORD_ID}")
print(f"PDF: {PDF_PATH} ({PDF_PATH.stat().st_size // 1024} KB)")

# Step 1: Create new version
print("\n[1/4] Creating new version draft...")
r = requests.post(f"{API}/records/{RECORD_ID}/versions", headers=HEADERS)
print(f"  Status: {r.status_code}")
if r.status_code not in [200, 201]:
    print(f"  Error: {r.text[:500]}")
    sys.exit(1)

draft = r.json()
draft_id = draft["id"]
print(f"  New draft ID: {draft_id}")

# Step 2: Delete old files
print("\n[2/4] Removing old files...")
files_url = f"{API}/records/{draft_id}/draft/files"
r = requests.get(files_url, headers=HEADERS)
if r.status_code == 200:
    for f in r.json().get("entries", []):
        key = f["key"]
        requests.delete(f"{files_url}/{key}", headers=HEADERS)
        print(f"  Deleted: {key}")

# Step 3: Upload new PDF
print("\n[3/4] Uploading new PDF...")
filename = "WhyLab - Causal Audit Framework for Stable Agent Self-Improvement.pdf"

# Initiate upload
r = requests.post(
    files_url,
    headers={**HEADERS, "Content-Type": "application/json"},
    data=json.dumps([{"key": filename}]),
)
print(f"  Initiate status: {r.status_code}")

# Upload content
with open(PDF_PATH, "rb") as fp:
    r = requests.put(
        f"{files_url}/{filename}/content",
        headers={**HEADERS, "Content-Type": "application/octet-stream"},
        data=fp,
    )
print(f"  Upload status: {r.status_code}")

# Commit file
r = requests.post(
    f"{files_url}/{filename}/commit",
    headers=HEADERS,
)
print(f"  Commit status: {r.status_code}")

# Step 4: Update metadata
print("\n[4/4] Updating metadata...")
metadata = {
    "metadata": {
        "title": "WhyLab: A Causal Audit Framework for Stable Agent Self-Improvement",
        "publication_date": "2026-03-17",
        "version": "2.0.0",
        "resource_type": {"id": "publication-preprint"},
        "description": (
            "<p>Self-improving AI agents risk cognitive policy oscillation "
            "when noisy feedback causes strategy degradation. "
            "<strong>WhyLab</strong> provides a causal safety monitoring framework "
            "with three components:</p>"
            "<ul>"
            "<li><strong>C1</strong>: Information-theoretic drift detection</li>"
            "<li><strong>C2</strong>: E-value sensitivity filter for fragile successes</li>"
            "<li><strong>C3</strong>: Lyapunov-bounded adaptive damping</li>"
            "</ul>"
            "<p>On SWE-bench Lite (300 problems), C2 maintains zero regressions "
            "across 10,500 episodes. A non-stationary experiment (E6) validates "
            "all three components independently: C3 reduces energy by 49%, "
            "C2 reduces oscillation by 76%, C1 cuts drift detection delay by 16%.</p>"
            "<p>Code: https://github.com/Yesol-Pilot/WhyLab</p>"
        ),
        "creators": [{"person_or_org": {"type": "personal", "given_name": "Anonymous", "family_name": "Author"}}],
    }
}

r = requests.put(
    f"{API}/records/{draft_id}/draft",
    headers={**HEADERS, "Content-Type": "application/json"},
    data=json.dumps(metadata),
)
print(f"  Metadata status: {r.status_code}")
if r.status_code not in [200, 201]:
    print(f"  Response: {r.text[:500]}")

print(f"\nDraft ready at: https://zenodo.org/records/{draft_id}")
print("Review and publish manually at Zenodo, or run with --publish flag.")
