"""Download annual RUNOFF and BASEFLOW NetCDF files from the CEC VIC historical simulation.

Source: https://wrf-cmip6-noversioning.s3.amazonaws.com/index.html
        #lusu/CEC/VIC_SIMULATIONS/GCMs/dfinal_historical/{RUNOFF,BASEFLOW}/

Files: 1951.nc – 2020.nc per variable (~534 MB each)

Total streamflow = RUNOFF (surface) + BASEFLOW (subsurface drainage).
"""

from __future__ import annotations

import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://wrf-cmip6-noversioning.s3.amazonaws.com"
VARIABLES = {
    "RUNOFF":   Path(__file__).parent / "runoff",
    "BASEFLOW": Path(__file__).parent / "baseflow",
}

CHUNK_SIZE = 1 * 1024 * 1024  # 1 MB read chunks
MAX_RETRIES = 5
RETRY_BACKOFF = 5  # seconds between retries
WORKERS = 2  # parallel downloads (keep low given ~534 MB per file)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def remote_size(url: str) -> int | None:
    """Return Content-Length of a remote file, or None if unavailable."""
    try:
        resp = requests.head(url, timeout=15)
        resp.raise_for_status()
        return int(resp.headers.get("Content-Length", 0)) or None
    except Exception:
        return None


def download_file(url: str, dest: Path) -> None:
    """Download *url* to *dest* with resume support and retry logic."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    existing = tmp.stat().st_size if tmp.exists() else 0

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            headers = {"Range": f"bytes={existing}-"} if existing else {}
            with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
                # 416 = Range Not Satisfiable → file already complete
                if resp.status_code == 416:
                    tmp.rename(dest)
                    return
                resp.raise_for_status()

                total = int(resp.headers.get("Content-Length", 0)) or None
                if total and existing:
                    total += existing  # adjust for already-downloaded bytes

                mode = "ab" if existing else "wb"
                with (
                    open(tmp, mode) as fh,
                    tqdm(
                        total=total,
                        initial=existing,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=dest.name,
                        leave=False,
                    ) as bar,
                ):
                    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                        fh.write(chunk)
                        bar.update(len(chunk))

            tmp.rename(dest)
            return

        except (requests.RequestException, OSError) as exc:
            existing = tmp.stat().st_size if tmp.exists() else 0
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * attempt
                print(
                    f"  [{dest.name}] attempt {attempt} failed ({exc}); "
                    f"retrying in {wait}s …",
                    file=sys.stderr,
                )
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Failed to download {url} after {MAX_RETRIES} attempts"
                ) from exc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _list_keys_for(prefix: str) -> list[str]:
    """Return all .nc S3 keys under *prefix*."""
    keys: list[str] = []
    continuation_token: str | None = None
    while True:
        params: dict[str, str] = {
            "list-type": "2",
            "prefix": prefix,
            "max-keys": "1000",
        }
        if continuation_token:
            params["continuation-token"] = continuation_token
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        for content in root.findall("s3:Contents", ns):
            key_el = content.find("s3:Key", ns)
            if key_el is not None and key_el.text:
                keys.append(key_el.text)
        truncated_el = root.find("s3:IsTruncated", ns)
        if truncated_el is None or truncated_el.text.lower() != "true":
            break
        token_el = root.find("s3:NextContinuationToken", ns)
        continuation_token = token_el.text if token_el is not None else None
    return [k for k in keys if k.endswith(".nc")]


def main() -> None:
    errors: list[str] = []

    for varname, out_dir in VARIABLES.items():
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"lusu/CEC/VIC_SIMULATIONS/GCMs/dfinal_historical/{varname}/"

        print(f"\nListing {varname} files …")
        keys = _list_keys_for(prefix)
        print(f"Found {len(keys)} files to download → {out_dir}")

        total_bar = tqdm(keys, desc=varname, unit="file")

        for key in total_bar:
            filename = Path(key).name
            dest = out_dir / filename
            url = f"{BASE_URL}/{key}"

            if dest.exists():
                expected = remote_size(url)
                if expected is None or dest.stat().st_size == expected:
                    total_bar.write(f"  skip  {filename} (already complete)")
                    continue
                else:
                    total_bar.write(
                        f"  re-download  {filename} "
                        f"(size mismatch: local={dest.stat().st_size} remote={expected})"
                    )

            total_bar.write(f"  downloading  {filename}")
            try:
                download_file(url, dest)
                total_bar.write(f"  done  {filename}")
            except Exception as exc:
                total_bar.write(f"  ERROR  {filename}: {exc}", file=sys.stderr)
                errors.append(f"{varname}/{filename}")

    if errors:
        print(f"\n{len(errors)} file(s) failed:", file=sys.stderr)
        for name in errors:
            print(f"  {name}", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
