#!/usr/bin/env python3
"""Serial single-GPU QSVT sweep over the OpenFOAM momentum matrices, abandoned at the first run that exceeds the time limit.

    python3 run_sweep.py --device lightning.gpu --limit 600 --out /work/results/sweep_gpu.json

Order: increasing N, then increasing kappa (cheap runs first). The first run that exceeds ``--limit`` seconds is killed and recorded as
TIMEOUT; every later matrix is recorded as NOT_RUN (sweep abandoned), as instructed. One run at a time, one process, one GPU.
"""
import argparse, glob, json, os, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="lightning.gpu")
ap.add_argument("--limit", type=float, default=600.0, help="seconds; the sweep is abandoned at the first run exceeding this")
ap.add_argument("--matrices", default="/work/matrices")
ap.add_argument("--glob", default="*_Ux_call*_A.npz")
ap.add_argument("--out", default="/work/results/sweep.json")
a = ap.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
items = []
for p in glob.glob(os.path.join(a.matrices, a.glob)):
    pre = p[: -len("_A.npz")]
    meta = json.load(open(pre + "_meta.json"))
    items.append((meta["N"], meta["kappa"], pre))
items.sort()

rows, abandoned = [], False
print(f"{len(items)} matrices, device={a.device}, limit={a.limit}s", flush=True)
for N, kappa, pre in items:
    name = os.path.basename(pre)
    if abandoned:
        rows.append(dict(matrix=name, N=N, kappa=kappa, status="NOT_RUN"))
        continue
    out = f"/work/results/{name}.{a.device}.json"
    if os.path.exists(out):
        os.remove(out)
    t0 = time.time()
    p = subprocess.Popen([sys.executable, os.path.join(here, "qsvt_solve_case.py"), "--matrix", pre, "--device", a.device, "--out", out],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        log, _ = p.communicate(timeout=a.limit)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, 9)
        p.communicate()
        wall = time.time() - t0
        rows.append(dict(matrix=name, N=N, kappa=kappa, status="TIMEOUT", wall_s=wall))
        print(f"TIMEOUT {name} (N={N}) after {wall:.0f}s -> sweep abandoned", flush=True)
        abandoned = True
        continue
    wall = time.time() - t0
    if p.returncode == 0 and os.path.exists(out):
        r = json.load(open(out))
        rows.append(dict(matrix=name, N=N, kappa=kappa, status="OK", wall_s=wall, degree=r["degree"], qubits=r["qubits"],
                         p_success=r["p_success"], rel_err_direction=r["rel_err_direction"], timings=r["timings"], device=r["device"],
                         visible_gpus=r.get("visible_gpus")))
        print(f"OK      {name} N={N} kappa={kappa:.2f} deg={r['degree']} err={r['rel_err_direction']:.4f} wall={wall:.1f}s", flush=True)
    else:
        rows.append(dict(matrix=name, N=N, kappa=kappa, status="FAILED", wall_s=wall, log_tail=log[-1500:]))
        print(f"FAILED  {name}: {log[-300:]}", flush=True)
    json.dump(rows, open(a.out, "w"), indent=1)
json.dump(rows, open(a.out, "w"), indent=1)
print("sweep finished; abandoned =", abandoned, flush=True)
