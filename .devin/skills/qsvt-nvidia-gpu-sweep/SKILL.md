---
name: qsvt-nvidia-gpu-sweep
description: Run the single-GPU QSVT (quantum singular value transform) linear-solve sweep on an NVIDIA GPU server (tested on B300 / compute capability 10.3) inside a container, without touching anything already on the server. Covers password SSH from Windows, read-only recon, reusing an existing image, persisting pip libraries outside the container, PennyLane lightning.gpu, the time-boxed sweep and result collection.
---

# QSVT single-GPU sweep on an NVIDIA server (PennyLane lightning.gpu)

Code, data and results: https://github.com/brandt-22/qsvt-b300-nvidia-experiments (private). AMD counterpart: skill `pennylane-amdgpu-qsvt`.

Reproduces the non-QIO experiment: for each OpenFOAM momentum matrix A (sparse, N = 1024 / 2304 / 4096), solve A x = b with QSVT on an emulated
statevector (dense block encoding, `PennyLane lightning.gpu`), compare with a classical solve, one run at a time on ONE GPU, abandon the sweep at
the first run that exceeds ~10 minutes. These are ideal, noiseless emulated qubits; a classical sparse solve takes milliseconds.

## Rules that were requested (keep them)
* Do not disturb anything on the server: read-only recon first, one new directory only, one container, no host installs, no changes to existing files/containers/GPUs in use.
* Do all work in a container. Libraries brought in go to a persistent directory OUTSIDE the container (mounted), so they survive it. Do not install anything that already exists on the machine/image; if a different version is needed, bring it into the container's mounted dir.
* Clean up: containers use `--rm`; verify `docker ps` is empty at the end. Leave the work dir (results) in place.

## Workflow
1. **Access (Windows, password SSH).** No sshpass; use PuTTY `plink.exe` / `pscp.exe`. First connect with `echo y | plink ...` to cache the host key, afterwards `-batch`.
   Keep the password in an env var (`$env:NVPW`), never in a file. PowerShell mangles nested quotes and `$`: **write remote commands to a local `.sh` file and run `plink -m file.sh`**; upload scripts with `pscp` then `sed -i 's/\r$//'` (CRLF breaks bash).
2. **Read-only recon:** `nvidia-smi` (memory used, compute apps), `docker ps`, `docker images`, `df -h`, `free -g`, `ls /root`, `who/last`, connectivity to pypi. Pick a GPU that is idle (0 MiB, no compute apps) and re-check right before launching.
3. **Reuse an existing image** instead of pulling one. Inspect candidates with `docker run --rm --entrypoint python3 IMG -c "import numpy, scipy, torch ..."`. Used `mlperf-nvidia:llama2_70b_lora-pyt` (Python 3.12.3, numpy 2.1.0, scipy 1.18.1, CUDA 13.4 runtime). Override the entrypoint.
4. **Persist only the missing libraries.** `pip install --target` re-installs dependencies even if present, so first ask pip what is missing, then install exactly that with `--no-deps`:
   * `docker run --rm ... IMG bash -c 'pip install --dry-run --quiet --report /tmp/r.json pennylane pennylane-lightning pennylane-lightning-gpu pyqsp pytest; python3 -c "import json;[print(i[\"metadata\"][\"name\"]+\"==\"+i[\"metadata\"][\"version\"]) for i in json.load(open(\"/tmp/r.json\"))[\"install\"]]"'`
   * Result (22 packages, 2.0 GB): pennylane 0.45.1, pennylane_lightning 0.45.0, pennylane_lightning_gpu 0.45.0, custatevec-cu12 1.15.0, nvidia-{cublas,cuda-runtime,cuda-nvrtc,cusparse,nvjitlink}-cu12 12.9.x, pyqsp 0.2.0, pytest, autograd, autoray, cachetools, diastatic-malt, rustworkx, scipy-openblas32, termcolor, tomlkit, appdirs, iniconfig, pluggy.
   * numpy and scipy come from the image and are NOT re-installed.
   * `pip install --no-deps --target /root/qsvt_b300/libs <pinned list>` inside a throwaway container with `-v /root/qsvt_b300/libs:/libs`.
5. **Container recipe** (`nv_container.sh`): `docker run --rm --gpus device=N --memory 512g --cpus 24 --pids-limit 4096 -v work:/work -v libs:/libs -e PYTHONPATH=/libs -e OMP_NUM_THREADS=8 -e LD_LIBRARY_PATH=/libs/nvidia/{cuda_runtime,cublas,cusparse,nvjitlink,cuda_nvrtc}/lib:/libs/cuquantum/lib ... --entrypoint bash IMG -c "..."`.
   The CUDA-12 wheels in `/libs` coexist with the image's CUDA 13 runtime because they come first on `LD_LIBRARY_PATH`. Resource caps keep a shared box safe. Only one GPU is visible inside (`nvidia-smi` shows index 0).
6. **Smoke test:** `qml.device("lightning.gpu", wires=24)` and `lightning.qubit`. **lightning.gpu works on B300 (sm_103, driver 580, CUDA 13.0)** with the CUDA-12 cuQuantum 1.15 wheel.
7. **Validate before sweeping:** matrix `case07_n32_nu0.01_U1_Ux_call11` must give degree 135, p_success 0.0205419, direction error 0.0058508 (identical to the earlier AMD MI300X run to ~12 digits; CPU path identical). B300: 19.6 s total vs 33.8 s on the CPU path.
8. **Sweep** (`run_sweep.py`): serial, ordered by increasing N then kappa, subprocess per matrix with `timeout=600`, kill the process group on timeout, record TIMEOUT, mark all remaining as NOT_RUN. Start it detached (`nohup ... &`, log to a file) and poll with short `plink` calls; each tool call has a ~5 minute ceiling.
9. **Collect** `results/sweep_gpu.json`, build the table, then copy results + new code (not `libs/`, not matrices that already exist elsewhere unless wanted) to the backup host.

## Code (what to copy; all new code is small)
`qsvt_lib.py` (inverse polynomial + pyqsp angles + dense unitary dilation), `qsvt_solve_case.py` (one matrix, one device), `run_sweep.py`, `smoke.py`, `nv_container.sh`.
The solver MUST block-encode **A^T** (QSVT with an odd polynomial inverts the adjoint side); encoding A silently solves A^T x = b (3-5% error on non-symmetric matrices, invisible on symmetric ones).
Full reference implementation and conventions: `cfd_qsvt_reference` (NumPy; includes a sparse banded-LCU block encoding).

## Gotchas
* `pip install --target` ignores what is installed -> always `--no-deps` with an explicit list from `--dry-run --report`.
* Run time on the dense emulation scales with polynomial degree (about linear): N=1024 about 1 s per 7 degrees; N=2304 (dim 8192) about 3 s per degree. The 10-minute cut falls at degree ~190-200 for N=2304; N=4096 is far beyond it.
* The pennylane image on the AMD host prints harmless warnings; here none were needed. Do not leave `docker run` without `--rm`.
* `plink` prints "Store key in cache?" on the first connection even with piped input; that is expected.
* No `git`/`gh` on the Windows workstation. Push from a host that has git and reaches GitHub (my AMD host did) with `GIT_ASKPASS` and the token piped over stdin (never in a URL, file or `.git/config`), or use the REST API. Commit as `Devin (automated)` to avoid exposing a work email; create new repos private by default.

## Results from the B300 run
See `RESULTS.md` next to this file.

## Gate-level and time-boxed sweeps (added after the B300 experiments 1-3)
Code, reports and results: https://github.com/brandt-22/qsvt-b300-nvidia-experiments (`gate_level/`, `reports/`).
* **Validate before any sweep** with small-N checks against a NumPy reference (`validate_gate.py`): they caught a wrong shift direction, a non-identity branch, and a mis-set tolerance (pyqsp reproduces the polynomial only to ~1e-4) before hours of runtime were spent.
* **The earlier AMD sparse script's `increment_circuit` is a decrement.** Increment = flip bit i MSB-first controlled on all lower bits; decrement = the reverse order. Only matters for non-symmetric matrices.
* **Real-part extraction with an extra ancilla** (H on f; PCPhase(phi) plus f-controlled PCPhase(-2 phi) after every block-encoding call; H on f) gives (s+ + s-)/2 in ONE circuit run, half the gates of two runs.
* **Exact per-entry coefficients cost a multiplexed rotation** (2^(index+system qubits) terms per call, Gray-code RY/CNOT built once and reused). Averaged coefficients are 100-300x cheaper on the GPU but solve a different matrix (7-53 % error on the OpenFOAM momentum matrices).
* **First-call GPU time is warm-up** (about 0.2 s); a second call of the same small circuit takes milliseconds. Report both. GPU time becomes the bottleneck only at about 28+ qubits (state vector size doubles per qubit); one B300 (275 GB) reached 34 qubits (256 GiB state vector).
* **Dense N-sweeps are CPU-bound** (fusing the QSVT unitary on the CPU is 80-99 % of the time); the host CPU/container CPU cap, not the accelerator, decides how far they get.
* **Tooling:** `pscp` takes ONE remote source per call; PowerShell does not support `<` redirection (upload helper scripts with `pscp`); write long jobs as a driver script, start it detached (`nohup`) and poll with short calls; run each sweep value as its own process with `timeout --signal=KILL` and stop the sweep at the first run over the limit.
