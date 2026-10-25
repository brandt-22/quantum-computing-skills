#!/usr/bin/env python3
"""AMD (MI300X host) vs NVIDIA (B300 server, B300) single-GPU QSVT comparison table. Reads raw per-run JSON; writes markdown + csv."""
import glob, json, os, re, sys
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
amd = {r["matrix"]: r for r in json.load(open(os.path.join(R, "amd_results_g1a.json")))}
nv = {}
for p in glob.glob(os.path.join(R, "case*Ux_call*.lightning.gpu.json")):
    r = json.load(open(p)); nv[r["matrix"]] = r
sweep = {re.sub(r"_A$", "", s["matrix"]): s for s in json.load(open(os.path.join(R, "sweep_gpu.json")))}
names = sorted(sweep, key=lambda n: (sweep[n]["N"], sweep[n]["kappa"], n))
hdr = ["Matrix", "N", "kappa", "Degree", "AMD MI300X s", "NVIDIA B300 s", "B300 speedup", "AMD error", "B300 error", "AMD p_succ", "B300 p_succ"]
rows = []
for n in names:
    s, a, b = sweep[n], amd.get(n), nv.get(n)
    short = n.replace("_Ux_", " ").replace("_nu", " nu").replace("_U", " U")
    t_amd = a["timings"]["total_s"] if a else None
    t_nv = b["timings"]["total_s"] if b else None
    if s["status"] == "TIMEOUT": nvs, err_nv, ps_nv, deg = "TIMEOUT >601", "", "", ""
    elif s["status"] == "NOT_RUN": nvs, err_nv, ps_nv, deg = "not run", "", "", ""
    else: nvs, err_nv, ps_nv, deg = f"{t_nv:.1f}", f"{b['rel_err_direction']:.4f}", f"{b['p_success']:.4f}", str(b["degree"])
    rows.append([short, str(s["N"]), f"{s['kappa']:.2f}", deg, f"{t_amd:.1f}" if a else "not run",
                 nvs, f"{t_amd / t_nv:.1f}x" if (a and t_nv) else "", f"{a['rel_err_direction']:.4f}" if a else "", err_nv,
                 f"{a['p_success']:.4f}" if a else "", ps_nv])
md = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"] + ["| " + " | ".join(r) + " |" for r in rows]
open(os.path.join(R, "comparison_amd_vs_b300.md"), "w").write("\n".join(md) + "\n")
open(os.path.join(R, "comparison_amd_vs_b300.csv"), "w").write("\n".join(",".join(c.replace(",", ";") for c in r) for r in [hdr] + rows) + "\n")
print("\n".join(md))
both = [(amd[n]["timings"]["total_s"], nv[n]["timings"]["total_s"]) for n in names if n in amd and n in nv and sweep[n]["status"] == "OK"]
print(f"\nrows with both: {len(both)}; speedup min {min(a / b for a, b in both):.1f}x, max {max(a / b for a, b in both):.1f}x, "
      f"sum AMD {sum(a for a, b in both):.0f}s vs B300 {sum(b for a, b in both):.0f}s = {sum(a for a, b in both) / sum(b for a, b in both):.1f}x")
