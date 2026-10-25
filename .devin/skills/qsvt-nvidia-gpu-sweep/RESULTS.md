# B300 single-GPU QSVT sweep: results

B300 server, 8x NVIDIA B300 SXM6 (compute capability 10.3, 275 GB each), driver 580.167.08. Used ONE GPU (index 0), inside container
`mlperf-nvidia:llama2_70b_lora-pyt` (existing image, nothing pulled) with PennyLane 0.45.1 + lightning.gpu 0.45.0 (cuStateVec 1.15.0, CUDA 12 wheels in
`/root/qsvt_b300/libs`), `OMP_NUM_THREADS=8`. Dense block encoding, ideal emulated qubits, serial runs, 600 s limit, sweep abandoned at the first timeout.

Outcome: 26 OK, 1 TIMEOUT (601 s), 1 NOT_RUN (sweep abandoned), 0 FAILED. Direction error vs classical solve 0.0035 to 0.0068 on all OK runs.
Full table: `table.txt` (tab-separated + Markdown). Raw data: `sweep_gpu.json`, per-case JSONs.

Validation (case07 n32 nu0.01 U1 call11, kappa 13.39): B300 lightning.gpu degree 135, p_success 0.0205419, error 0.0058508, 19.6 s total;
CPU path (lightning.qubit, same container) identical numbers, 33.8 s. The earlier AMD MI300X run gave the same degree, p_success and error to ~12 digits (68.6 s total
on that host, different CPU/thread settings, so the time ratio is indicative only).

Observations
* Run time is roughly linear in polynomial degree. N=1024 (11 qubits): ~0.15 s/degree. N=2304 and N=4096 both use 13 qubits (statevector dimension 8192; N=2304 is padded), at ~3 s/degree, so for the same degree N=4096 costs the same as N=2304.
* The 10-minute limit is crossed at degree ~200+: N=2304 kappa 19.8 (degree 201) took 587 s (just under); N=4096 kappa 26.2 (degree ~265) timed out at 601 s.
* The first run in a fresh process pays GPU initialisation (first QSVT sequence 3.8 s vs ~1.1 s afterwards at N=1024).
* A classical sparse solve of any of these systems takes milliseconds; this is an emulation/validation exercise, not a speed claim.
