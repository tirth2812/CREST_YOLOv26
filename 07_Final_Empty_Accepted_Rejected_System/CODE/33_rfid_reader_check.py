"""
CREST - STANDALONE RFID READER CHECK  (READ ONLY)

* Only READS the PLC.  It never writes anything: no piston, no conveyor.
* Checks RFID1 and RFID2 independently, no camera / vision involved.

What to do
  1. Run it.
  2. Pass pallets 1..6 by RFID1, then by RFID2 (stopped, then moving at the
     real conveyor speed).
  3. Watch the checklist.  Press Ctrl+C for the summary.

What it shows
  * every logical read: ONE line when a tag appears (no repeats while it stays)
  * how long each tag stayed in the reader field ("dwell")  <- tells us whether a
    MOVING pallet is seen for longer than one PLC poll
  * the checklist of pallets 1..6 per reader
  * PLC poll rate / read time (timestamp resolution for the vision matching)
"""

from pylogix import PLC
import time

PLC_IP = "192.168.82.231"

STATIONS = {
    "RFID1": [
        "IO_Link_Master_ST1_to_ST4:I1.Data[96]",
        "IO_Link_Master_ST1_to_ST4:I1.Data[97]",
        "IO_Link_Master_ST1_to_ST4:I1.Data[98]",
        "IO_Link_Master_ST1_to_ST4:I1.Data[99]",
    ],
    "RFID2": [
        "IO_Link_Master_ST1_to_ST4:I1.Data[128]",
        "IO_Link_Master_ST1_to_ST4:I1.Data[129]",
        "IO_Link_Master_ST1_to_ST4:I1.Data[130]",
        "IO_Link_Master_ST1_to_ST4:I1.Data[131]",
    ],
}

RFID_MAP = {
    (-8188, 336, -18499, -14824): 1,
    (-8188, 336, -18498, 7750): 2,
    (-8188, 336, -18499, 21259): 3,
    (-8188, 336, -18498, 11138): 4,
    (-8188, 336, -18498, 5934): 5,
    (-8188, 336, -18499, 12969): 6,
}

NO_TAG = (0, 0, 0, 0)

POLL_SLEEP_SECONDS = 0.005
STILL_PRESENT_WARNING_SECONDS = 10.0

ALL_TAGS = [tag for tags in STATIONS.values() for tag in tags]


def clock():
    return time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"


def read_batch(comm):
    """One PLC round trip for all 8 words.  Returns {station: tuple} or None."""

    responses = comm.Read(ALL_TAGS)

    values = {}
    index = 0

    for station, tags in STATIONS.items():
        words = []

        for _ in tags:
            response = responses[index]
            index += 1

            if response.Status != "Success" or response.Value is None:
                return None

            words.append(response.Value)

        values[station] = tuple(words)

    return values


def read_single(comm):
    """Fallback (like the old demo): one round trip per word."""

    values = {}

    for station, tags in STATIONS.items():
        words = []

        for tag in tags:
            response = comm.Read(tag)

            if response.Status != "Success" or response.Value is None:
                return None

            words.append(response.Value)

        values[station] = tuple(words)

    return values


def checklist_line(station, tally):
    parts = []

    for pallet in range(1, 7):
        count = tally[station].get(pallet, 0)
        parts.append(f"{pallet}:{'x' + str(count) if count else '-'}")

    return f"{station}  " + "  ".join(parts)


def print_summary(tally, unknown, dwell, polls, read_times, errors, mode):
    print()
    print("=" * 68)
    print("SUMMARY")
    print("=" * 68)

    for station in STATIONS:
        missing = [p for p in range(1, 7) if not tally[station].get(p)]
        print(checklist_line(station, tally))
        print(
            "   -> all pallets 1-6 read correctly"
            if not missing
            else f"   -> NOT YET SEEN: {missing}"
        )

    if unknown:
        print()
        print("UNKNOWN TAG VALUES (not in RFID_MAP):")
        for station, raw in sorted(unknown):
            print(f"   {station}: {raw}")

    print()
    print("DWELL (how long a tag stayed in the field, from first to last poll that saw it)")
    for station in STATIONS:
        values = sorted(dwell[station])

        if values:
            print(
                f"   {station}: n={len(values)}  min={values[0]:.3f}s  "
                f"median={values[len(values) // 2]:.3f}s  max={values[-1]:.3f}s"
            )
        else:
            print(f"   {station}: no reads")

    print()
    print(f"PLC polling mode: {mode}")

    if read_times:
        mean = sum(read_times) / len(read_times)
        print(
            f"   polls={polls}  read time mean={mean * 1000:.1f} ms  "
            f"max={max(read_times) * 1000:.1f} ms  errors={errors}"
        )
        print(f"   => the PLC value is sampled about every {(mean + POLL_SLEEP_SECONDS) * 1000:.0f} ms")

    print("=" * 68)


def main():
    tally = {station: {} for station in STATIONS}
    unknown = set()
    dwell = {station: [] for station in STATIONS}

    # per station: None or {"raw", "first", "last", "polls", "warned"}
    present = {station: None for station in STATIONS}

    polls = 0
    errors = 0
    read_times = []
    mode = None
    last_error = None

    print("=" * 68)
    print(" RFID READER CHECK - READ ONLY (no piston, no writes)")
    print(" Pass pallets 1-6 by RFID1, then by RFID2.  Ctrl+C = summary.")
    print("=" * 68)

    comm = PLC()
    comm.IPAddress = PLC_IP
    comm.SocketTimeout = 5

    try:
        while True:
            started = time.monotonic()

            try:
                if mode in (None, "one read for all 8 words"):
                    values = read_batch(comm)

                    if values is not None and mode is None:
                        mode = "one read for all 8 words"

                    if values is None and mode is None:
                        # batch not usable here: fall back to the demo's style
                        values = read_single(comm)

                        if values is not None:
                            mode = "one read per word (fallback)"
                            print("NOTE: batch read not accepted by this PLC path - using single reads (slower).")
                else:
                    values = read_single(comm)

                if values is None:
                    raise RuntimeError("PLC read returned a non-success status")

            except KeyboardInterrupt:
                raise

            except Exception as error:
                errors += 1

                if str(error) != last_error:
                    print(f"[{clock()}] PLC READ ERROR: {error}")
                    last_error = str(error)

                time.sleep(0.5)
                continue

            now = time.monotonic()
            polls += 1
            read_times.append(now - started)
            last_error = None

            for station, raw in values.items():
                state = present[station]

                # ---------- tag leaves / changes ----------
                if state is not None and raw != state["raw"]:
                    seconds = state["last"] - state["first"]
                    dwell[station].append(seconds)
                    print(
                        f"[{clock()}] {station}  tag left the field   "
                        f"dwell={seconds:.3f}s  ({state['polls']} polls)"
                    )
                    present[station] = None
                    state = None

                if raw == NO_TAG:
                    continue

                # ---------- same tag still there ----------
                if state is not None:
                    state["last"] = now
                    state["polls"] += 1

                    if (
                        not state["warned"]
                        and now - state["first"] >= STILL_PRESENT_WARNING_SECONDS
                    ):
                        state["warned"] = True
                        print(
                            f"[{clock()}] {station}  same tag still in the field after "
                            f"{STILL_PRESENT_WARNING_SECONDS:.0f}s (pallet stopped?)"
                        )
                    continue

                # ---------- a NEW tag appeared: ONE logical event ----------
                present[station] = {
                    "raw": raw,
                    "first": now,
                    "last": now,
                    "polls": 1,
                    "warned": False,
                }

                pallet = RFID_MAP.get(raw)

                if pallet is None:
                    unknown.add((station, raw))
                    print(f"[{clock()}] {station}  UNKNOWN TAG  raw={raw}")
                    continue

                tally[station][pallet] = tally[station].get(pallet, 0) + 1

                print(f"[{clock()}] {station}  PALLET {pallet}  raw={raw}")
                print("            " + checklist_line(station, tally))

            time.sleep(POLL_SLEEP_SECONDS)

    except KeyboardInterrupt:
        pass

    # a tag still in the field when we stop
    for station, state in present.items():
        if state is not None:
            dwell[station].append(state["last"] - state["first"])

    print_summary(tally, unknown, dwell, polls, read_times, errors, mode or "n/a")


if __name__ == "__main__":
    main()
