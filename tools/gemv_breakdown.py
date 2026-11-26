#!/usr/bin/env python3
"""
Splits a decode profile's GEMV time by projection, to show which weight
matrices stream at close to full memory bandwidth and which do not.

`make profile-qwen` runs this after Nsight Systems has written
decode_profile.sqlite. By hand:

    python3 tools/gemv_breakdown.py [profile.sqlite] [weights_prefix]
    defaults: decode_profile.sqlite  models/qwen

Every decode step launches the GEMV 169 times in a fixed order: q, k, v, o,
gate, up and down for each of the 24 layers, then the LM head. So each launch
is labelled by its position in that sequence. Prefill runs on the GEMM, so
every GEMV launch in the profile belongs to a decode step. fp32 and fp16
weights run different instantiations of the kernel, told apart by name.
"""

import sqlite3
import statistics
import sys
from collections import defaultdict

LAYER_PROJECTIONS = ["q", "k", "v", "o", "gate", "up", "down"]
ALL_PROJECTIONS = LAYER_PROJECTIONS + ["lm_head"]


def read_config(prefix):
    cfg = {}
    with open(prefix + ".idx") as f:
        for line in f:
            parts = line.split()
            if parts and parts[0] == "config":
                cfg[parts[1]] = float(parts[2])
    return cfg


def main():
    db = sys.argv[1] if len(sys.argv) > 1 else "decode_profile.sqlite"
    prefix = sys.argv[2] if len(sys.argv) > 2 else "models/qwen"

    c = read_config(prefix)
    H, I, V = int(c["hidden"]), int(c["intermediate"]), int(c["vocab"])
    layers = int(c["layers"])
    QD = int(c["heads"] * c["head_dim"])
    KVD = int(c["kv_heads"] * c["head_dim"])
    # (K, N) of each weight matrix, stored (in x out).
    shape = {"q": (H, QD), "k": (H, KVD), "v": (H, KVD), "o": (QD, H),
             "gate": (H, I), "up": (H, I), "down": (I, H), "lm_head": (H, V)}
    per_step = {p: layers for p in LAYER_PROJECTIONS}
    per_step["lm_head"] = 1
    order = LAYER_PROJECTIONS * layers + ["lm_head"]  # one decode step

    con = sqlite3.connect(db)
    peak = None
    try:
        row = con.execute("SELECT memoryBandwidth FROM TARGET_INFO_GPU LIMIT 1").fetchone()
        if row and row[0] and row[0] > 1e9:
            peak = row[0] / 1e9  # bytes/s -> GB/s
    except sqlite3.Error:
        pass

    columns = {r[1] for r in con.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    extras = [x for x in ("gridX", "registersPerThread", "localMemoryPerThread",
                          "staticSharedMemory") if x in columns]
    query = ("SELECT k.start, k.end, s.value" + "".join(f", k.{x}" for x in extras) +
             " FROM CUPTI_ACTIVITY_KIND_KERNEL AS k"
             " JOIN StringIds AS s ON k.demangledName = s.id"
             " WHERE s.value LIKE '%gemv_kernel%' ORDER BY k.start")
    launches = con.execute(query).fetchall()
    if not launches:
        sys.exit(f"No GEMV launches found in {db}")

    runs = defaultdict(list)  # weight type -> launches, in time order
    for launch in launches:
        runs["fp16" if "half" in launch[2] else "fp32"].append(launch)

    print("\nGEMV time per decode step, by projection")
    if peak:
        print(f"(peak memory bandwidth from the profile: {peak:.0f} GB/s)")

    step_us_by_type = {}
    for wtype in ("fp32", "fp16"):
        rows = runs.get(wtype)
        if not rows:
            continue
        bytes_per_weight = 2 if wtype == "fp16" else 4
        if len(rows) % len(order) != 0:
            print(f"\nwarning: {len(rows)} {wtype} launches is not a whole number of "
                  f"{len(order)}-launch steps; labels may be off")

        durations = defaultdict(list)  # projection -> microseconds per launch
        grid = {}
        for i, launch in enumerate(rows):
            p = order[i % len(order)]
            durations[p].append((launch[1] - launch[0]) / 1e3)
            if "gridX" in extras:
                grid[p] = launch[3 + extras.index("gridX")]

        # A step on the GPU timeline: from one step's first GEMV to the next
        # step's. The median skips the gaps where a prefill sits between reps.
        firsts = [rows[i][0] for i in range(0, len(rows), len(order))]
        gaps = [(b - a) / 1e3 for a, b in zip(firsts, firsts[1:])]
        step_us = statistics.median(gaps) if gaps else float("nan")

        info = dict(zip(extras, rows[0][3:]))
        details = []
        if "registersPerThread" in info:
            details.append(f"{info['registersPerThread']} registers/thread")
        if "localMemoryPerThread" in info:
            details.append(f"{info['localMemoryPerThread']} B local memory/thread")
        if "staticSharedMemory" in info:
            details.append(f"{info['staticSharedMemory']} B shared memory/block")
        print(f"\n{wtype} weights: {len(rows) // len(order)} decode steps"
              + (f"; {', '.join(details)}" if details else ""))
        print(f"  {'projection':<10} {'K x N':>13} {'blocks':>7} {'us/launch':>10} "
              f"{'GB/s':>7} {'% peak':>7} {'us/step':>9} {'share':>6}")

        total = 0.0
        rows_out = []
        for p in ALL_PROJECTIONS:
            med = statistics.median(durations[p])
            k, n = shape[p]
            gbs = k * n * bytes_per_weight / (med * 1e3)
            us = med * per_step[p]
            total += us
            rows_out.append((p, k, n, grid.get(p, ""), med, gbs, us))
        for p, k, n, g, med, gbs, us in rows_out:
            pct = f"{100 * gbs / peak:6.0f}%" if peak else f"{'':>7}"
            print(f"  {p:<10} {f'{k} x {n}':>13} {g:>7} {med:10.1f} {gbs:7.0f} {pct} "
                  f"{us:9.0f} {100 * us / total:5.1f}%")
        weight_bytes = sum(shape[p][0] * shape[p][1] * per_step[p] for p in ALL_PROJECTIONS)
        all_gbs = weight_bytes * bytes_per_weight / (total * 1e3)
        pct = f"{100 * all_gbs / peak:6.0f}%" if peak else f"{'':>7}"
        print(f"  {'all GEMVs':<10} {'':>13} {'':>7} {'':>10} {all_gbs:7.0f} {pct} {total:9.0f}")
        print(f"  decode step on the GPU timeline: {step_us:.0f} us, of which GEMV "
              f"{100 * total / step_us:.0f}% and everything else {step_us - total:.0f} us")
        step_us_by_type[wtype] = {p: statistics.median(durations[p]) for p in ALL_PROJECTIONS}
        step_us_by_type[wtype]["_step"] = step_us
        step_us_by_type[wtype]["_gemv"] = total

    if len(step_us_by_type) == 2:
        a, b = step_us_by_type["fp32"], step_us_by_type["fp16"]
        print("\nfp16 speedup over fp32, per launch:")
        print("  " + "  ".join(f"{p} {a[p] / b[p]:.2f}x" for p in ALL_PROJECTIONS))
        print(f"  all GEMVs {a['_gemv'] / b['_gemv']:.2f}x, "
              f"whole decode step {a['_step'] / b['_step']:.2f}x")


if __name__ == "__main__":
    main()
