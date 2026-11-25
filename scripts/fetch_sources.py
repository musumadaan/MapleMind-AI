"""Download the ORIGINAL public sources behind data/kb/*.md for review and KB refresh.

  python scripts/fetch_sources.py            # saves raw HTML/PDF to data/raw/<doc_id>.* and a manifest with sha256 + date

The KB files are concise, attributed summaries so the demo runs offline. For production, re-generate chunks from these
raw sources (keep the front-matter metadata), and re-run `maplemind ingest`. Respect each publisher's licence:
Statistics Canada Open Licence, Government of Canada terms, CMAJ/PMC open-access terms, CIHI terms of use.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"


def main():
    import requests
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = []
    for md in sorted((ROOT / "data/kb").glob("*.md")):
        meta = yaml.safe_load(md.read_text().split("---")[1])
        url = meta.get("url")
        if not url:
            continue
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "MapleMind-research/0.1"})
            ext = ".pdf" if "pdf" in r.headers.get("content-type", "") else ".html"
            out = RAW / f"{meta['doc_id']}{ext}"
            out.write_bytes(r.content)
            manifest.append({"doc_id": meta["doc_id"], "url": url, "status": r.status_code, "bytes": len(r.content),
                             "sha256": hashlib.sha256(r.content).hexdigest(), "fetched": time.strftime("%Y-%m-%d")})
            print(f"ok   {meta['doc_id']:<40} {r.status_code}")
        except Exception as e:  # noqa: BLE001
            manifest.append({"doc_id": meta["doc_id"], "url": url, "error": repr(e)})
            print(f"fail {meta['doc_id']:<40} {e!r}", file=sys.stderr)
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
