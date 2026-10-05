"""
Phase B/C/D helper: summarise a rfid_fusion_*.jsonl log written by
29_live_phase27_rfid_fusion.py.

    python analyze_rfid_fusion_log.py path\\to\\rfid_fusion_YYYYmmdd_HHMMSS.jsonl
"""

import json
import statistics
import sys
from collections import Counter, defaultdict


def describe(values):
    if not values:
        return "n=0"
    ordered = sorted(values)
    p = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
    return (
        f"n={len(values)} min={ordered[0] * 1000:+.0f}ms p10={p(0.10) * 1000:+.0f}ms "
        f"median={statistics.median(values) * 1000:+.0f}ms p90={p(0.90) * 1000:+.0f}ms "
        f"max={ordered[-1] * 1000:+.0f}ms"
    )


def main(path):
    records = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    counts = Counter(r["event"] for r in records)

    print("EVENT COUNTS")
    for name, count in sorted(counts.items()):
        print(f"  {name:32s} {count}")

    matches = [r for r in records if r["event"] == "RFID_PASSAGE_MATCH"]
    per_station = defaultdict(lambda: defaultdict(list))
    for r in matches:
        st = r["station"]
        per_station[st]["rfid_minus_entry"].append(r["rfid_minus_entry"])
        if r.get("rfid_minus_exit") is not None:
            per_station[st]["rfid_minus_exit"].append(r["rfid_minus_exit"])
        per_station[st]["residual"].append(r["residual"])
        per_station[st]["resolve_latency"].append(r["resolve_latency"])

    for station, groups in sorted(per_station.items()):
        print(f"\n{station}  (use rfid_minus_entry / rfid_minus_exit to choose pre/post margins)")
        for key, values in groups.items():
            print(f"  {key:18s} {describe(values)}")

    print("\nHEALTH")
    opened = counts["PASSAGE_OPEN"]
    expired = counts["PASSAGE_EXPIRED"]
    print(f"  passages opened={opened} exited={counts['PASSAGE_EXIT']} expired={expired}"
          + (f" ({100 * expired / opened:.0f}% expired - high => bad geometry/continuity)" if opened else ""))
    print(f"  RFID events={counts['RFID_EVENT']} matched={counts['RFID_PASSAGE_MATCH']} "
          f"unmatched={counts['RFID_UNMATCHED']} ambiguous={counts['RFID_PASSAGE_AMBIGUOUS']} "
          f"debounced={counts['RFID_DEBOUNCE_DROP']} stale={counts['RFID_STALE_DROP']}")
    print(f"  tracker switches inside a passage={counts['PASSAGE_TRACK_SWITCH']} "
          f"station overlaps={counts['STATION_OVERLAP']}")
    print(f"  identity: assign={counts['IDENTITY_ASSIGN']} verify={counts['IDENTITY_VERIFY']} "
          f"remap={counts['IDENTITY_REMAP']} clear={counts['IDENTITY_CLEAR']} "
          f"unbound={counts['IDENTITY_UNBOUND']} reacquire={counts['IDENTITY_REACQUIRE']} "
          f"reacquire_ambiguous={counts['IDENTITY_REACQUIRE_AMBIGUOUS']}")

    remaps = [r for r in records if r["event"] == "IDENTITY_REMAP"]
    if remaps:
        print("\n  REMAPS (each one means vision believed a different pallet - inspect them):")
        for r in remaps[:20]:
            print(f"    t={r.get('t')} {r['station']} track={r['track_id']} {r['old_physical']} -> {r['new_physical']}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
