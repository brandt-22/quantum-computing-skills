#!/usr/bin/env python3
"""Solve A x = b for one OpenFOAM momentum matrix with QSVT on an emulated statevector.

    python3 qsvt_solve_case.py --matrix /work/matrices/<case>_Ux_call11 [--device lightning.gpu|lightning.qubit] [--out result.json]

The emulated device runs the QSVT sequence (block-encoding unitary applied `degree` times, interleaved with projector-controlled phases)
and the +phi / -phi sequences are combined to extract the real part of the block (LCU equivalent). Compared against a classical direct solve.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import qsvt_lib as ql

ap = argparse.ArgumentParser()
ap.add_argument("--matrix", required=True, help="prefix of <prefix>_A.npz / _b.npy / _x_ref.npy / _meta.json")
ap.add_argument("--device", default="lightning.gpu")
ap.add_argument("--out", default="")
ap.add_argument("--rel_tol", type=float, default=1e-2, help="relative error target of the inverse polynomial on [1/kappa,1]")
a = ap.parse_args()

import pennylane as qml

T = {}
t_all = time.time()
A = sp.load_npz(a.matrix + "_A.npz").toarray()
b = np.load(a.matrix + "_b.npy")
x_ref = np.load(a.matrix + "_x_ref.npy")
meta = json.load(open(a.matrix + "_meta.json"))
N = A.shape[0]
nwires = int(np.ceil(np.log2(2 * N)))
dim = 2 ** nwires

# --- normalise: A / alpha has singular values in [1/kappa, 1] ---
t0 = time.time()
sv = np.linalg.svd(A, compute_uv=False)
kappa = float(sv[0] / sv[-1])
alpha = float(sv[0]) * 1.001
cheb, degree = ql.inverse_poly(kappa * 1.05, rel_tol=a.rel_tol)
angles = ql.qsvt_angles(cheb)
T["classical_prep_s"] = time.time() - t0

# --- block encoding (dense unitary dilation, padded to 2^n) ---
t0 = time.time()
# An odd polynomial acts on the *adjoint* side of the block-encoded matrix (QSVT returns P^(SV)(M^T) for real M), so to apply A^-1 we
# block-encode M = A^T.  (Verified with qsvt_orientation_check.py: encoding A gave 3.3% error vs A^-1 b; encoding A^T gives 0.6%.)
U_small, _ = ql.block_encode_unitary(A.T, alpha)        # 2N x 2N
U = np.eye(dim, dtype=complex)
U[: 2 * N, : 2 * N] = U_small
T["block_encode_s"] = time.time() - t0

wires = list(range(nwires))
bhat = b / np.linalg.norm(b)
psi0 = np.zeros(dim, dtype=complex)
psi0[:N] = bhat
Pi = np.zeros(dim)            # projector onto the "A" subspace = first N basis states
Pi[:N] = 1.0


def qsvt_state(sign):
    """Run the QSVT phase sequence with angles*sign on the emulated device; return the final statevector."""
    dev = qml.device(a.device, wires=nwires)

    @qml.qnode(dev)
    def circ():
        qml.StatePrep(psi0, wires=wires)
        phases = sign * np.asarray(angles)
        d = len(phases) - 1
        qml.PCPhase(phases[0], dim=N, wires=wires)
        for k in range(1, len(phases)):
            qml.QubitUnitary(U if k % 2 == 1 else U.conj().T, wires=wires)
            qml.PCPhase(phases[k], dim=N, wires=wires)
        return qml.state()
    return np.asarray(circ())


# --- run on the emulated device ---
t0 = time.time()
s_plus = qsvt_state(+1)
T["qsvt_plus_s"] = time.time() - t0
t0 = time.time()
s_minus = qsvt_state(-1)
T["qsvt_minus_s"] = time.time() - t0

s = 0.5 * (s_plus + s_minus)          # real-part extraction (LCU of +phi and -phi), emulated
blk = s[:N]                           # amplitudes on the A subspace  ~  P(A^T-ish) |b>
p_success = float(np.linalg.norm(blk) ** 2)

xref_hat = x_ref / np.linalg.norm(x_ref)
cands = {}
for name, v in (("blk", blk), ("blk_real", blk.real)):
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
        # global sign / phase fixed by overlap with the classical solution
        ph = np.vdot(xref_hat, v)
        v = v * np.exp(-1j * np.angle(ph))
        cands[name] = float(np.linalg.norm(v - xref_hat))
x_rel_err = min(cands.values()) if cands else float("nan")
T["total_s"] = time.time() - t_all

# what a pure classical solve gives for reference timing
t0 = time.time(); _ = spla.spsolve(sp.csc_matrix(A), b); T["classical_spsolve_s"] = time.time() - t0

import subprocess
try:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,compute_cap", "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
except Exception as e:
    gpu = f"ERR {e}"
res = dict(matrix=os.path.basename(a.matrix), N=N, qubits=nwires, kappa=kappa, degree=degree, device=a.device,
           p_success=p_success, rel_err_direction=x_rel_err, timings=T,
           visible_gpus=gpu, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), pennylane=qml.__version__,
           host=os.uname().nodename)
print(json.dumps(res, indent=1))
if a.out:
    json.dump(res, open(a.out, "w"))
