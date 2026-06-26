#!/usr/bin/env python3
"""Robust downloader for the CederGroupHub text-mined synthesis dataset.

Direct `curl` downloads from raw.githubusercontent.com can stall on this server.
This script uses HTTP range requests, retries each chunk, and can resume from an
existing partial file.
"""

from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


URLS = {
    "solid-state-2020": "https://raw.githubusercontent.com/CederGroupHub/text-mined-synthesis_public/master/solid-state_dataset_20200713.json.xz",
    "solid-state-2019": "https://raw.githubusercontent.com/CederGroupHub/text-mined-synthesis_public/master/solid-state_dataset_2019-12-03.json.xz",
    "sol-gel-2020": "https://raw.githubusercontent.com/CederGroupHub/text-mined-synthesis_public/master/sol-gel_dataset_20200713.json.xz",
}

API_CONTENTS = {
    "solid-state-2020": "https://api.github.com/repos/CederGroupHub/text-mined-synthesis_public/contents/solid-state_dataset_20200713.json.xz",
    "solid-state-2019": "https://api.github.com/repos/CederGroupHub/text-mined-synthesis_public/contents/solid-state_dataset_2019-12-03.json.xz",
    "sol-gel-2020": "https://api.github.com/repos/CederGroupHub/text-mined-synthesis_public/contents/sol-gel_dataset_20200713.json.xz",
}


def head_size(url: str, timeout: int) -> int:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return int(response.headers["Content-Length"])


def fetch_range(url: str, start: int, end: int, timeout: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Range": f"bytes={start}-{end}",
            "Accept-Encoding": "identity",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def download(url: str, output: Path, chunk_size: int, retries: int, timeout: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    total = head_size(url, timeout)
    current = output.stat().st_size if output.exists() else 0
    if current > total:
        output.unlink()
        current = 0
    print(f"target={output}")
    print(f"size={total}")
    print(f"resume_from={current}")
    with output.open("ab") as handle:
        while current < total:
            end = min(current + chunk_size - 1, total - 1)
            last_error: Exception | None = None
            for attempt in range(1, retries + 1):
                try:
                    data = fetch_range(url, current, end, timeout)
                    expected = end - current + 1
                    if len(data) != expected:
                        raise IOError(f"short chunk: got {len(data)}, expected {expected}")
                    handle.write(data)
                    handle.flush()
                    current += len(data)
                    print(f"{current}/{total}", flush=True)
                    break
                except (urllib.error.URLError, TimeoutError, OSError) as exc:
                    last_error = exc
                    wait = min(2 * attempt, 10)
                    print(f"retry chunk {current}-{end}, attempt {attempt}/{retries}: {exc}")
                    time.sleep(wait)
            else:
                raise RuntimeError(f"failed chunk {current}-{end}") from last_error
    if output.stat().st_size != total:
        raise RuntimeError(f"download incomplete: {output.stat().st_size} != {total}")


def download_via_github_api(dataset: str, output: Path, timeout: int) -> None:
    """Download through GitHub API metadata/blob endpoints.

    This path is useful when `raw.githubusercontent.com` HEAD works but streaming
    file content stalls. It first obtains the blob URL through the contents API.
    """

    output.parent.mkdir(parents=True, exist_ok=True)
    contents_url = API_CONTENTS[dataset]
    metadata_request = urllib.request.Request(
        contents_url,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(metadata_request, timeout=timeout) as response:
        metadata = json.loads(response.read().decode("utf-8"))
    blob_url = metadata["git_url"]
    expected_size = int(metadata["size"])
    blob_request = urllib.request.Request(
        blob_url,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(blob_request, timeout=timeout) as response:
        blob = json.loads(response.read().decode("utf-8"))
    content = base64.b64decode(blob["content"])
    output.write_bytes(content)
    if output.stat().st_size != expected_size:
        raise RuntimeError(f"download incomplete: {output.stat().st_size} != {expected_size}")
    print(f"wrote {output} ({output.stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(URLS), default="solid-state-2020")
    parser.add_argument("--url", default="")
    parser.add_argument("--output", type=Path, default=Path("raw/solid-state_dataset_20200713.json.xz"))
    parser.add_argument("--chunk-size", type=int, default=128 * 1024)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=25)
    parser.add_argument("--method", choices=["range", "api"], default="range")
    args = parser.parse_args()
    url = args.url or URLS[args.dataset]
    if args.method == "api":
        download_via_github_api(args.dataset, args.output, args.timeout)
    else:
        download(url, args.output, args.chunk_size, args.retries, args.timeout)


if __name__ == "__main__":
    main()
