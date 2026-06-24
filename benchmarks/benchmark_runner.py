# Copyright (C) 2025 Senior AI Architect.
# Licensed under AGPLv3. For commercial licensing, contact the author.
#
# Single entrypoint for the KSDZ benchmark suite.
#
#   python benchmarks/benchmark_runner.py --all --output results.json
#
# Two metric classes are kept strictly separate (see ksdz_adapter.py for the
# rationale):
#   * lossless        -> zlib/zstd/lz4. Metric: compression ratio only.
#                        Exact round-trip is REQUIRED and asserted.
#   * lossy_transform -> KSDZ. Metrics: RMSE/NRMSE, L2/L-inf, spectral
#                        distortion (raw + normalized). Its size ratio is
#                        recorded as a *pseudo* ratio, flagged non-comparable.
# A guard refuses to rank the two classes on one leaderboard. The execution
# environment is captured in the output so numbers are reproducible.

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import baselines
import datasets as ds
import ksdz_adapter
import metrics as mx

DEFAULT_SIZES = [4096, 65536]


def _pkg_version(name):
    try:
        return getattr(__import__(name), "__version__", "unknown")
    except Exception:
        return None


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_HERE,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def capture_environment():
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            "numpy": _pkg_version("numpy"),
            "zstandard": _pkg_version("zstandard"),
            "lz4": _pkg_version("lz4"),
        },
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _base_row(dataset, codec, cls):
    return {
        "dataset": dataset.name,
        "codec": codec.name,
        "class": cls,
        "params": getattr(codec, "params", {}),
        "N": dataset.N,
        "seed": dataset.seed,
        "input_sha256": dataset.sha256,
        "original_size": dataset.original_size,
    }


def run_lossless(dataset, codec):
    """Run a lossless baseline. Asserts exact round-trip — a lossless codec that
    does not reconstruct bit-for-bit is a bug, recorded as status 'error'."""
    row = _base_row(dataset, codec, "lossless")
    row.update({"cr_comparable": True, "lossless": True})
    if not getattr(codec, "available", False):
        row["status"] = "unavailable"
        row["reason"] = getattr(codec, "reason", "dependency missing")
        return row
    try:
        t0 = time.perf_counter()
        comp = codec.compress(dataset.data)
        t1 = time.perf_counter()
        recon = codec.decompress(comp)
        t2 = time.perf_counter()
        exact = (recon == dataset.data)
        if not exact:
            row["status"] = "error"
            row["reason"] = "lossless round-trip violated (output != input)"
            return row
        row.update({
            "status": "ok",
            "reason": None,
            "compressed_size": len(comp),
            "compression_ratio": mx.compression_ratio(dataset.original_size, len(comp)),
            "roundtrip_ok": True,
            "encode_seconds": t1 - t0,
            "decode_seconds": t2 - t1,
        })
    except Exception as e:
        row["status"] = "error"
        row["reason"] = repr(e)
    return row


def run_lossy(dataset, codec):
    """Run a lossy transform. No exact-reconstruction assertion; fidelity is the
    point of measurement. The size ratio is a pseudo ratio, flagged."""
    row = _base_row(dataset, codec, "lossy_transform")
    row.update({"cr_comparable": False, "lossless": False})
    if not getattr(codec, "available", False):
        row["status"] = "unavailable"
        row["reason"] = getattr(codec, "reason", "dependency missing")
        return row
    try:
        t0 = time.perf_counter()
        enc = codec.encode(dataset.data)
        t1 = time.perf_counter()
        recon = codec.decode(enc)
        t2 = time.perf_counter()
        fidelity = mx.reconstruction_metrics(dataset.array, recon)
        row.update({
            "status": "ok",
            "reason": None,
            "compressed_size": len(enc),
            "pseudo_compression_ratio": mx.compression_ratio(dataset.original_size, len(enc)),
            "encode_seconds": t1 - t0,
            "decode_seconds": t2 - t1,
            **fidelity,
        })
    except Exception as e:
        row["status"] = "error"
        row["reason"] = repr(e)
    return row


def assert_no_class_mixing(report):
    """Epistemic gate: every result must carry a class, and a lossless result
    must never expose a comparable ratio under the same key as a lossy one.
    lossless -> compression_ratio (cr_comparable True);
    lossy    -> pseudo_compression_ratio (cr_comparable False)."""
    for r in report["results"]:
        assert r.get("class") in ("lossless", "lossy_transform"), \
            f"result missing valid class: {r.get('codec')}"
        if r["class"] == "lossless":
            assert "pseudo_compression_ratio" not in r, \
                "lossless result leaked a pseudo ratio"
        else:
            assert "compression_ratio" not in r, \
                "lossy result exposed a comparable compression_ratio"


def run_suite(dataset_names, sizes, seed, top_k):
    lossless = baselines.build_lossless_codecs()
    lossy = ksdz_adapter.build_lossy_codecs(top_k=top_k)
    results = []
    for name in dataset_names:
        for N in sizes:
            dataset = ds.build_dataset(name, N, seed)
            for codec in lossless:
                results.append(run_lossless(dataset, codec))
            for codec in lossy:
                results.append(run_lossy(dataset, codec))
    report = {
        "schema_version": "1.0",
        "environment": capture_environment(),
        "config": {
            "seed": seed,
            "sizes": sizes,
            "datasets": dataset_names,
            "top_k": top_k,
            "codecs": [c.name for c in lossless] + [c.name for c in lossy],
        },
        "results": results,
    }
    assert_no_class_mixing(report)
    return report


def print_summary(report):
    rows = report["results"]
    print("\n== LOSSLESS BASELINES (metric: compression ratio; exact round-trip) ==")
    print(f"{'dataset':<20}{'codec':<12}{'N':>8}{'ratio':>9}{'enc_s':>9}{'dec_s':>9}  status")
    for r in rows:
        if r["class"] != "lossless":
            continue
        if r["status"] != "ok":
            print(f"{r['dataset']:<20}{r['codec']:<12}{r.get('N',0):>8}"
                  f"{'-':>9}{'-':>9}{'-':>9}  {r['status']}")
            continue
        print(f"{r['dataset']:<20}{r['codec']:<12}{r['N']:>8}"
              f"{r['compression_ratio']:>9.4f}{r['encode_seconds']:>9.4f}"
              f"{r['decode_seconds']:>9.4f}  ok")

    print("\n== LOSSY TRANSFORM (KSDZ) — fidelity, NOT comparable to lossless ratio ==")
    print(f"{'dataset':<20}{'codec':<14}{'N':>8}{'nrmse':>9}{'Linf':>7}"
          f"{'spec_norm':>11}{'~ratio':>9}{'enc_s':>9}  status")
    for r in rows:
        if r["class"] != "lossy_transform":
            continue
        if r["status"] != "ok":
            print(f"{r['dataset']:<20}{r['codec']:<14}{r.get('N',0):>8}"
                  f"{'-':>9}{'-':>7}{'-':>11}{'-':>9}{'-':>9}  {r['status']}")
            continue
        print(f"{r['dataset']:<20}{r['codec']:<14}{r['N']:>8}"
              f"{r['nrmse']:>9.4f}{r['linf_error']:>7.0f}"
              f"{r['spectral_distortion_normalized']:>11.4f}"
              f"{r['pseudo_compression_ratio']:>9.4f}{r['encode_seconds']:>9.4f}  ok")


def main(argv=None):
    p = argparse.ArgumentParser(description="KSDZ reproducible benchmark suite.")
    p.add_argument("--all", action="store_true",
                   help="Run every registered dataset (default if --datasets omitted).")
    p.add_argument("--datasets", nargs="+", choices=ds.dataset_names(),
                   help="Subset of datasets to run.")
    p.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES,
                   help=f"Signal lengths N (default {DEFAULT_SIZES}).")
    p.add_argument("--seed", type=int, default=1234, help="Master seed (default 1234).")
    p.add_argument("--top-k", type=int, default=100, help="KSDZ top_k (default 100).")
    p.add_argument("--output", default="results.json", help="Output JSON path.")
    args = p.parse_args(argv)

    names = args.datasets if args.datasets else ds.dataset_names()
    report = run_suite(names, args.sizes, args.seed, args.top_k)

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print_summary(report)
    print(f"\n[*] Wrote {len(report['results'])} results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
