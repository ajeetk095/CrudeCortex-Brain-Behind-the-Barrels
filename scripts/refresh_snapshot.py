"""Refresh data/fred/*.csv from FRED's public CSV endpoint and rewrite data/fred/MANIFEST.json.

Usage (needs internet):  python scripts/refresh_snapshot.py
No API key is needed. Existing files are only replaced after the download validates.
"""
import json, os, sys
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import config as C
from core import data_sources as DS
from core.exports import sha256_file


def write_manifest(refreshed=False):
    man = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "note": "Retrieval date of the original bundled files is unknown; checksums identify the files in use.",
           "files": {}}
    for sid in C.SERIES_BY_ID:
        p = DS.snapshot_path(sid)
        s, st = DS.parse_fred_csv_text(open(p).read(), sid)
        man["files"][sid] = {"sha256": sha256_file(p), "rows": int(len(s)), "first": str(s.index[0].date()),
                             "last": str(s.index[-1].date()), "refreshed_this_run": refreshed}
    json.dump(man, open(os.path.join(C.SNAPSHOT_DIR, "MANIFEST.json"), "w"), indent=2)


if __name__ == "__main__":
    if "--manifest-only" in sys.argv:
        write_manifest(); print("manifest written"); sys.exit(0)
    for sid in C.SERIES_BY_ID:
        text = DS.fetch_fred_csv(sid)
        DS.parse_fred_csv_text(text, sid)
        DS._atomic_write(DS.snapshot_path(sid), text)
        print("refreshed", sid)
    write_manifest(refreshed=True)
