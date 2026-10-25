#!/usr/bin/env python3
"""Build a copy-paste table (tab-separated and Markdown) from a sweep_gpu.json produced by run_sweep.py.

    python make_table.py sweep_gpu.json > table.txt
"""
import json, sys

rows = json.load(open(sys.argv[1]))
hdr = ["Matrix", "N", "Qubits", "kappa", "Degree", "Status", "Wall s", "QSVT +phi s", "QSVT -phi s", "p_success", "Direction error"]


def cells(r):
    t = r.get("timings", {})
    ok = r["status"] == "OK"
    return [r["matrix"].replace("_Ux_", " Ux ").replace("_A", ""), str(r["N"]), str(r.get("qubits", "")), f"{r['kappa']:.2f}",
            str(r.get("degree", "")), r["status"] + (" (>%d s)" % round(r["wall_s"]) if r["status"] == "TIMEOUT" else ""),
            f"{r['wall_s']:.1f}" if "wall_s" in r else "", f"{t['qsvt_plus_s']:.1f}" if ok else "", f"{t['qsvt_minus_s']:.1f}" if ok else "",
            f"{r['p_success']:.4f}" if ok else "", f"{r['rel_err_direction']:.4f}" if ok else ""]


body = [cells(r) for r in rows]
print("TAB-SEPARATED (paste into Excel/Sheets)")
print("\t".join(hdr))
for c in body:
    print("\t".join(c))
print("\nMARKDOWN")
print("| " + " | ".join(hdr) + " |")
print("|" + "|".join(["---"] * len(hdr)) + "|")
for c in body:
    print("| " + " | ".join(c) + " |")
done = [r for r in rows if r["status"] == "OK"]
print(f"\n{len(done)} OK, {sum(r['status']=='TIMEOUT' for r in rows)} TIMEOUT, {sum(r['status']=='NOT_RUN' for r in rows)} NOT_RUN (sweep abandoned), "
      f"{sum(r['status']=='FAILED' for r in rows)} FAILED")
if done:
    print(f"direction error range over OK runs: {min(r['rel_err_direction'] for r in done):.4f} to {max(r['rel_err_direction'] for r in done):.4f}")
