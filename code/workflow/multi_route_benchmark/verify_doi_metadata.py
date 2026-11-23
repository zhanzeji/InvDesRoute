#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "doi_metadata_cache"


def fetch(doi: str) -> dict[str, object]:
    cache_path = CACHE / f"{doi.replace('/', '__')}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
    result: dict[str, object] = {"doi": doi, "crossref_status": "unresolved"}
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=30, headers={"User-Agent": "InvDesRoute-literature-audit/1.0"})
            if response.status_code == 200:
                message = response.json()["message"]
                dates = message.get("published-print") or message.get("published-online") or message.get("issued") or {}
                date_parts = dates.get("date-parts", [[]])
                result = {
                    "doi": doi,
                    "crossref_status": "resolved",
                    "crossref_doi": message.get("DOI", ""),
                    "title": "; ".join(message.get("title", [])),
                    "container_title": "; ".join(message.get("container-title", [])),
                    "publisher": message.get("publisher", ""),
                    "publication_year": date_parts[0][0] if date_parts and date_parts[0] else "",
                    "type": message.get("type", ""),
                    "url": message.get("URL", ""),
                }
                break
            result["http_status"] = response.status_code
        except Exception as exc:
            result["error"] = str(exc)
        time.sleep(1.0 + attempt)
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    return result


def main() -> None:
    dataset = pd.read_csv(ROOT / "multi_route_literature_dataset.csv")
    dois = sorted(dataset["doi"].dropna().astype(str).unique())
    metadata = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fetch, doi): doi for doi in dois}
        for index, future in enumerate(as_completed(futures), start=1):
            metadata.append(future.result())
            if index % 10 == 0 or index == len(dois):
                print(f"Crossref {index}/{len(dois)}", flush=True)
    metadata_df = pd.DataFrame(metadata).sort_values("doi")
    metadata_df.to_csv(ROOT / "results" / "doi_metadata_audit.csv", index=False)
    joined = dataset.merge(metadata_df, on="doi", how="left")
    joined.to_csv(ROOT / "results" / "multi_route_literature_dataset_with_doi_metadata.csv", index=False)
    unresolved = metadata_df[metadata_df["crossref_status"] != "resolved"]
    unresolved.to_csv(ROOT / "results" / "unresolved_dois.csv", index=False)
    print(metadata_df["crossref_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
