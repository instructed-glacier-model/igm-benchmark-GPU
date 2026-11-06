#!/usr/bin/env python3
"""
IGM GPU benchmark: ice-flow time per step and GPU memory vs grid size, for two cases (float32):

  online_f32   network trained on the fly (1 Adam iteration per step)
  offline_f32  pretrained network dahunet_mini, inference only

Each (case, grid) is one `igm_run` on one idle GPU. Results go to results/<label>/ (results.csv, meta.json),
raw runs to outputs/<label>/. After an out-of-memory run, the finer grids of that case are skipped.

  ./run_bench.py --quick     # smoke test: 2 coarse grids, 20 yr
  ./run_bench.py             # full benchmark
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent

CASES = {"online_f32": "bench_online", "offline_f32": "bench_offline"}
BASE_NX, BASE_NY, BASE_DX = 9020, 6020, 100.0      # data/input_bench.nc
DEFAULT_RATIOS = [16, 8, 4, 3, 2]                  # 1600 m ... 200 m
QUICK_RATIOS, QUICK_YEARS = [16, 8], 20

OOM_PAT = re.compile(r"ResourceExhausted|out of memory|OOM when allocating|CUDA_ERROR_OUT_OF_MEMORY|"
                     r"hipErrorOutOfMemory|HIP_ERROR_OUT_OF_MEMORY", re.I)
COLS = ["case", "ratio", "dx_m", "nx", "ny", "cells_M", "status", "n_steps",
        "step_s", "iceflow_s", "iceflow_p10_s", "iceflow_p90_s", "Mcells_per_s",
        "first_step_s", "tf_peak_gib", "smi_peak_gib", "wall_s"]

VERSIONS = """
import os, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf, keras, igm
from importlib.metadata import version
b = tf.sysconfig.get_build_info()
g = tf.config.list_physical_devices("GPU")
d = tf.config.experimental.get_device_details(g[0]) if g else {}
print(json.dumps(dict(tensorflow=tf.__version__, keras=keras.__version__, cuda=b.get("cuda_version"),
                      cudnn=b.get("cudnn_version"), rocm=(open("/opt/rocm/.info/version").read().strip() if b.get("is_rocm_build") and os.path.exists("/opt/rocm/.info/version") else None), igm_version=version("igm-model"),
                      igm_path=os.path.dirname(igm.__file__), tf_gpu=d.get("device_name"),
                      compute_capability=d.get("compute_capability"))))
"""


def sh(cmd: list[str], timeout: float = 30, **kw) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw).stdout.strip()
    except Exception:
        return ""


def smi(gpu: int, query: str) -> str:
    return sh(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits", "-i", str(gpu)])


def amd_dev(gpu: int) -> Path | None:
    """sysfs device of the gpu-th AMD GPU (amdgpu driver, PCI order), None if there is none."""
    devs = sorted({d.resolve() for d in Path("/sys/class/drm").glob("card[0-9]*/device")
                   if (d / "mem_info_vram_used").exists()})
    return devs[gpu] if gpu < len(devs) else None


def amd_read(gpu: int, name: str) -> str:
    d = amd_dev(gpu)
    try:
        return (d / name).read_text().strip() if d else ""
    except OSError:
        return ""


def amd_name(gpu: int) -> str:
    name = _amd_name(gpu)
    return name if not name or name.startswith("AMD") else "AMD " + name


def _amd_name(gpu: int) -> str:
    try:
        return json.loads(sh(["amd-smi", "static", "-g", str(gpu), "-a", "--json"]))[
            "gpu_data"][0]["asic"]["market_name"]
    except Exception:
        m = re.search(r"Card Series:\s*(.+)", sh(["rocm-smi", "-d", str(gpu), "--showproductname"]))
        return m.group(1).strip() if m else ""


def smi_used_mib(gpu: int) -> int | None:
    """None where neither nvidia-smi nor amdgpu sysfs reports it (e.g. unified memory)."""
    try:
        return int(smi(gpu, "memory.used").splitlines()[0])
    except Exception:
        pass
    try:
        return int(amd_read(gpu, "mem_info_vram_used")) // 2**20
    except ValueError:
        return None


class SmiSampler(threading.Thread):
    """Peak of the GPU used memory (MiB, nvidia-smi or amdgpu sysfs), polled every 0.5 s."""

    def __init__(self, gpu: int):
        super().__init__(daemon=True)
        self.gpu, self.peak = gpu, None
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            v = smi_used_mib(self.gpu)
            if v is not None:
                self.peak = max(self.peak or 0, v)
            self._stop_evt.wait(0.5)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=5)
        return self.peak


def run_env(gpu: int) -> dict:
    return dict(os.environ, TF_USE_LEGACY_KERAS="0", TF_CPP_MIN_LOG_LEVEL="3", PYTHONUNBUFFERED="1",
                CUDA_VISIBLE_DEVICES=str(gpu), HIP_VISIBLE_DEVICES=str(gpu))


def collect_meta(args) -> dict:
    meta = {"label": args.label, "date": datetime.now().astimezone().isoformat(timespec="seconds"),
            "machine": platform.machine(), "cpu_count": os.cpu_count(), "gpu_index": args.gpu,
            "cases": args.cases, "ratios": args.ratios, "years": args.years}
    env = {k: os.environ[k] for k in ("TF_CUDNN_WORKSPACE_LIMIT_IN_MB", "XLA_FLAGS", "MIOPEN_FIND_MODE")
           if k in os.environ}
    if env:
        meta["env"] = env
    q = smi(args.gpu, "name,memory.total,driver_version")
    if q:
        name, mem, driver = [p.strip() for p in q.split(",")][:3]
        meta.update(gpu_name=name, gpu_mem=f"{mem} MiB", driver=driver)
    elif amd_dev(args.gpu):
        total = amd_read(args.gpu, "mem_info_vram_total")
        meta.update(gpu_name=amd_name(args.gpu), gpu_mem=f"{int(total) // 2**20} MiB" if total else None,
                    driver="amdgpu " + Path("/sys/module/amdgpu/version").read_text().strip()
                    if Path("/sys/module/amdgpu/version").exists() else "amdgpu")
    out = sh([sys.executable, "-c", VERSIONS], timeout=300, env=run_env(args.gpu))
    try:
        meta.update(json.loads(out.splitlines()[-1]))
        meta["igm_commit"] = sh(["git", "-C", meta.pop("igm_path"), "describe", "--tags", "--always", "--dirty"])
    except Exception:
        meta["env_error"] = "could not import tensorflow / igm"
    return meta


def body(xs):
    """Steps 2..N: step 1 carries the XLA compilation."""
    return xs[1:] if len(xs) > 1 else xs


def quantile(xs, q):
    xs = sorted(body(xs))
    return xs[min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))] if xs else float("nan")


def run_one(case: str, ratio: int, args, out_root: Path) -> dict:
    nx, ny, dx = BASE_NX // ratio, BASE_NY // ratio, BASE_DX * ratio
    name = f"{case}_r{ratio:02d}"
    run_dir, log = out_root / name, out_root / f"{name}.log"
    cmd = ["igm_run", f"+experiment={CASES[case]}", f"inputs.local.coarsening.ratio={ratio}",
           f"processes.time.end={args.years}", f"hydra.run.dir={run_dir}"]
    print(f"\n[bench] {case:<12} {dx:>5.0f} m  {nx} x {ny} ({nx * ny / 1e6:.2f} M cells)", flush=True)
    if args.dry_run:
        print("        " + " ".join(cmd))
        return {}
    shutil.rmtree(run_dir, ignore_errors=True)

    sampler = SmiSampler(args.gpu)
    sampler.start()
    t0 = time.monotonic()
    timed_out = False
    with log.open("w") as f:
        try:
            rc = subprocess.run(cmd, cwd=HERE, env=run_env(args.gpu), stdout=f, stderr=subprocess.STDOUT,
                                timeout=args.timeout_min * 60 or None).returncode
        except subprocess.TimeoutExpired:
            rc, timed_out = -9, True
    wall = time.monotonic() - t0
    smi_peak = sampler.stop()

    oom = OOM_PAT.search(log.read_text(errors="replace")[-30000:]) is not None
    status = "ok" if rc == 0 else "TIMEOUT" if timed_out else "OOM" if oom else "FAIL"
    try:
        b = json.loads((run_dir / "bench.json").read_text())
    except Exception:
        b = {}
    step, flow = b.get("step_s", []), b.get("iceflow_s", [])
    if status == "ok" and len(step) < 3:
        status = "FAIL"
    ok, nan = status == "ok", float("nan")
    iceflow = statistics.median(body(flow)) if ok and flow else nan
    peak = b.get("tf_peak_mib")
    row = {
        "case": case, "ratio": ratio, "dx_m": b.get("dx", dx), "nx": b.get("nx", nx), "ny": b.get("ny", ny),
        "status": status, "n_steps": len(step),
        "step_s": statistics.median(body(step)) if ok else nan,
        "iceflow_s": iceflow,
        "iceflow_p10_s": quantile(flow, 0.1) if ok else nan,
        "iceflow_p90_s": quantile(flow, 0.9) if ok else nan,
        "first_step_s": step[0] if step else nan,
        "tf_peak_gib": peak / 1024 if peak is not None else nan,
        "smi_peak_gib": smi_peak / 1024 if smi_peak is not None else nan,
        "wall_s": wall,
    }
    row["cells_M"] = row["nx"] * row["ny"] / 1e6
    row["Mcells_per_s"] = row["cells_M"] / iceflow if iceflow == iceflow and iceflow > 0 else nan
    print(f"        -> {status}  ice flow {iceflow * 1e3:.1f} ms/step  TF peak {row['tf_peak_gib']:.2f} GiB  "
          f"({row['n_steps']} steps, {wall:.0f} s, log {log.relative_to(HERE)})", flush=True)
    return row


def write_csv(rows, path: Path):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows({k: f"{r[k]:.6g}" if isinstance(r[k], float) else r[k] for k in COLS} for r in rows)
    tmp.replace(path)


def default_label(gpu: int) -> str:
    name = smi(gpu, "name") or (amd_dev(gpu) and amd_name(gpu)) or "gpu"
    name = re.sub(r"^NVIDIA\s+|GeForce\s+|^AMD\s+|Radeon\s+|\s+(Workstation|Server) Edition$", "", name)
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=int, default=0, help="GPU index (default 0)")
    p.add_argument("--label", help="results/<label>/ (default: GPU name, e.g. rtx_4090)")
    p.add_argument("--cases", default=",".join(CASES), help="among %(default)s")
    p.add_argument("--ratios", help=f"coarsening ratios of the 100 m grid, coarse first (default {DEFAULT_RATIOS})")
    p.add_argument("--years", type=float, help="model years per run, 1-yr steps (default 100)")
    p.add_argument("--quick", action="store_true", help=f"smoke test: ratios {QUICK_RATIOS}, {QUICK_YEARS} yr")
    p.add_argument("--timeout-min", type=float, default=60, help="kill a run after N min (0 = never)")
    p.add_argument("--idle-mib", type=int, default=1000,
                   help="the GPU counts as idle below this used memory (MiB, default %(default)s)")
    p.add_argument("--force", action="store_true", help="run even if the GPU is not idle")
    p.add_argument("--dry-run", "-n", action="store_true", help="print the runs only")
    args = p.parse_args()

    args.cases = [c for c in args.cases.split(",") if c]
    if set(args.cases) - set(CASES):
        sys.exit(f"unknown case in {args.cases}; choose among {list(CASES)}")
    args.ratios = ([int(r) for r in args.ratios.split(",")] if args.ratios
                   else QUICK_RATIOS if args.quick else DEFAULT_RATIOS)
    args.years = args.years or (QUICK_YEARS if args.quick else 100.0)
    args.label = args.label or default_label(args.gpu) + ("_quick" if args.quick else "")

    used = smi_used_mib(args.gpu)
    if used is not None and used > args.idle_mib and not (args.force or args.dry_run):
        sys.exit(f"GPU {args.gpu} already uses {used} MiB: the benchmark needs an idle GPU (--force to override).")

    res_dir, out_root = HERE / "results" / args.label, HERE / "outputs" / args.label
    rows, kept = [], []          # kept: rows of an earlier call that are not rerun now
    if (res_dir / "results.csv").exists() and not args.dry_run:
        with (res_dir / "results.csv").open() as f:
            kept = [r for r in csv.DictReader(f)
                    if not (r["case"] in args.cases and int(r["ratio"]) in args.ratios)]
    if not args.dry_run:
        res_dir.mkdir(parents=True, exist_ok=True)
        out_root.mkdir(parents=True, exist_ok=True)
        meta = collect_meta(args)
        (res_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        print(f"[bench] {meta.get('gpu_name', '?')} | TF {meta.get('tensorflow', '?')} | "
              f"IGM {meta.get('igm_version', '?')} -> {res_dir.relative_to(HERE)}")

    for case in args.cases:
        for r in args.ratios:
            row = run_one(case, r, args, out_root)
            if not row:
                continue
            rows.append(row)
            write_csv(kept + rows, res_dir / "results.csv")
            if row["status"] in ("OOM", "TIMEOUT"):
                print(f"[bench] {row['status']}: finer grids of {case} skipped")
                break
    if rows:
        print(f"\n[bench] done -> {(res_dir / 'results.csv').relative_to(HERE)}")


if __name__ == "__main__":
    main()
