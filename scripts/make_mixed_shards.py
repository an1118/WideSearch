#!/usr/bin/env python3
"""Generate NGPU balanced, en+zh-mixed instance_id shards for WideSearch.

Each shard (lane) gets an even total (~TOTAL/NGPU) and mixes English and
Chinese tasks. Writes one comma-separated instance_id list per line
(line i -> lane i), consumed by scripts/run_widesearch_parallel.sh via
LANE_IDS_FILE. Also prints a per-lane summary to stderr.

Balancing: English IDs are distributed round-robin starting at lane 0;
Chinese IDs round-robin starting at lane (len(en) %% ngpu), so the two
remainders land on disjoint lanes and per-lane totals stay within 1 of
each other. Within a lane, en/zh are interleaved so a partially-run lane
still covers both languages.

Usage:
    python scripts/make_mixed_shards.py --ngpu 8 --out /tmp/ws_mixed_shards.txt
    # offline / no HF:
    python scripts/make_mixed_shards.py --ngpu 8 --synthetic --n-en 100 --n-zh 100
"""
import argparse
import sys
from itertools import zip_longest


def get_ids_from_loader():
    from src.evaluation.data_loader import WideSearchDataLoaderHF

    dl = WideSearchDataLoaderHF()
    return list(dl.get_instance_id_list())


def get_ids_synthetic(n_en, n_zh):
    en = [f"ws_en_{i:03d}" for i in range(1, n_en + 1)]
    zh = [f"ws_zh_{i:03d}" for i in range(1, n_zh + 1)]
    return en + zh


def build_shards(ids, ngpu):
    en = sorted(i for i in ids if "_en_" in i)
    zh = sorted(i for i in ids if "_zh_" in i)
    lanes = [[] for _ in range(ngpu)]

    # English round-robin from lane 0.
    for k, iid in enumerate(en):
        lanes[k % ngpu].append(iid)

    # Chinese round-robin starting where English's remainder ended, so the
    # two remainders occupy disjoint lanes and totals stay balanced.
    zh_start = len(en) % ngpu
    for k, iid in enumerate(zh):
        lanes[(zh_start + k) % ngpu].append(iid)

    # Interleave en/zh within each lane.
    for li in range(ngpu):
        lane_en = [x for x in lanes[li] if "_en_" in x]
        lane_zh = [x for x in lanes[li] if "_zh_" in x]
        mixed = []
        for a, b in zip_longest(lane_en, lane_zh):
            if a is not None:
                mixed.append(a)
            if b is not None:
                mixed.append(b)
        lanes[li] = mixed
    return lanes, en, zh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngpu", type=int, default=8)
    ap.add_argument("--out", type=str, default="/tmp/ws_mixed_shards.txt")
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="build ids as ws_en_001..N + ws_zh_001..M instead of querying the "
        "HF loader (offline fallback)",
    )
    ap.add_argument("--n-en", type=int, default=100)
    ap.add_argument("--n-zh", type=int, default=100)
    args = ap.parse_args()

    if args.synthetic:
        ids = get_ids_synthetic(args.n_en, args.n_zh)
        src = f"synthetic ({args.n_en} en + {args.n_zh} zh)"
    else:
        ids = get_ids_from_loader()
        src = "ByteDance-Seed/WideSearch loader"

    lanes, en, zh = build_shards(ids, args.ngpu)

    with open(args.out, "w") as f:
        for lane in lanes:
            f.write(",".join(lane) + "\n")

    # Summary to stderr (stdout stays clean for scripting).
    print(f"[shards] source: {src}", file=sys.stderr)
    print(
        f"[shards] total={len(ids)} en={len(en)} zh={len(zh)} ngpu={args.ngpu}",
        file=sys.stderr,
    )
    print(f"[shards] wrote {args.out}", file=sys.stderr)
    total_assigned = 0
    for li, lane in enumerate(lanes):
        n_en = sum(1 for x in lane if "_en_" in x)
        n_zh = sum(1 for x in lane if "_zh_" in x)
        total_assigned += len(lane)
        head = ",".join(lane[:4])
        print(
            f"  lane {li}: {len(lane):2d} tasks ({n_en} en + {n_zh} zh)  e.g. {head}...",
            file=sys.stderr,
        )
    assert total_assigned == len(ids), (
        f"assigned {total_assigned} != {len(ids)} — sharding dropped tasks"
    )
    print(f"[shards] all {total_assigned} tasks assigned, none dropped/duplicated", file=sys.stderr)


if __name__ == "__main__":
    main()
