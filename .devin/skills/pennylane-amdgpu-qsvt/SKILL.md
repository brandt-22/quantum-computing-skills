---
name: pennylane-amdgpu-qsvt
description: Build and benchmark QSVT (Quantum Singular Value Transformation) circuits with PennyLane on AMD Instinct GPUs, including sparsity-aware LCU block encodings
argument-hint: "[matrix description or N]"
allowed-tools:
  - read
  - write
  - edit
  - exec
  - grep
  - glob
triggers:
  - user
  - model
---

# PennyLane QSVT on AMD Instinct GPUs (lightning.amdgpu)

This skill captures a working methodology for running QSVT circuits on AMD GPUs
via PennyLane's `lightning.amdgpu` backend, and ??? critically ??? how to build a
**genuinely sparsity-aware block encoding** instead of relying on PennyLane's
generic `qml.BlockEncode`, which does not exploit matrix sparsity and causes
build time to blow up superlinearly for anything beyond small dense matrices.

## Building the environment

`lightning.amdgpu` is not pip-installable as a prebuilt wheel for all
architectures ??? for MI300/MI355X-class GPUs (gfx942/gfx950) it typically needs
to be built from source against the target GPU architecture. A working
Dockerfile pattern:

```dockerfile
FROM rocm/dev-ubuntu-24.04:7.1
RUN apt-get update && apt-get install -y python3-venv python3-dev python3-pip \
    cmake ninja-build git libomp-dev rocthrust-dev rocprim-dev
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && pip install pennylane numpy scipy
RUN git clone --depth 1 --branch v0.45.0 https://github.com/PennyLaneAI/pennylane-lightning.git /opt/pennylane-lightning
WORKDIR /opt/pennylane-lightning
RUN pip install cmake ninja pybind11 build setuptools wheel
# lightning_qubit must be built first (it's a dependency of the other backends)
RUN PL_BACKEND="lightning_qubit" python scripts/configure_pyproject_toml.py && pip install . -vv
ENV CMAKE_ARGS="-DCMAKE_CXX_COMPILER=hipcc -DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_AMD_GFX950=ON -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_CXX_STANDARD=20 -DGPU_TARGETS=gfx950"
RUN PL_BACKEND="lightning_amdgpu" python scripts/configure_pyproject_toml.py && pip install . -vv
```

Adjust `Kokkos_ARCH_AMD_GFX950` / `GPU_TARGETS=gfx950` to match the actual GPU
(e.g. `GFX942`/`gfx942` for MI300X). Building takes 10-20+ minutes.

Run containers with:
```
docker run --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=<gpu_index> ...
```
See the `rocm-multi-gpu-pinning` skill for a critical gotcha about GPU selection
env vars ??? do not set both `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES`.

## The core lesson: BlockEncode is NOT sparsity-aware

`qml.BlockEncode(A, wires)` builds a generic embedding formula from the full
matrix values regardless of how many entries are zero. Its PennyLane
decomposition does not automatically produce a shallow circuit for a sparse
matrix ??? build time (classical preprocessing + circuit fusion) scales
superlinearly with matrix dimension `N`, becoming impractical (minutes to
hours) well before N reaches tens of thousands, even for a matrix that is
mostly zeros.

**If your matrix has real structure (sparse, banded, from a discretized PDE),
build your own LCU-based block encoding that reflects that structure** ??? this
is the only way to get the sparsity advantage QSVT is theoretically supposed
to offer.

## Building a sparsity-aware LCU block encoding for a banded/shift-structured matrix

For a matrix expressible as a sum of a diagonal term plus shifted-identity
terms (e.g. `A = 2I - S+ - S-` for a 1D discretized Laplacian, or the 2D
generalization `A = 4I - S+_row - S-_row - S+_col - S-_col` for a 5-point
stencil), use a Linear Combination of Unitaries (LCU) construction:

1. **Efficient shift circuits.** A cyclic "+1 mod 2^n" shift on an n-qubit
   register can be built in O(n) gates (not O(2^n) like an arbitrary dense
   unitary):
   ```python
   def increment_circuit(wires):
       n = len(wires)
       for i in range(n - 1, -1, -1):
           controls = wires[i + 1:]
           if controls:
               qml.MultiControlledX(wires=list(controls) + [wires[i]], control_values=[1] * len(controls))
           else:
               qml.PauliX(wires=wires[i])
   # decrement_circuit is the same gates in reverse order (each gate is self-inverse)
   ```

2. **LCU PREPARE + SELECT + PREPARE-dagger.** Use `k` ancilla qubits to select
   among the terms (`ceil(log2(num_terms))`), with `qml.StatePrep` to load
   `sqrt(|c_i| / sum|c_i|)` amplitudes, a `qml.DiagonalQubitUnitary` on the
   ancilla to imprint the correct sign pattern (do NOT try to compose this from
   PauliZ tensor products unless you've worked out the exact factorization ???
   `DiagonalQubitUnitary` with an explicit `[+1,-1,-1,...]` array is simpler and
   always correct), then `qml.ctrl(...)` the shift circuits on the ancilla
   basis states, then `qml.adjoint` of the PREPARE step.

3. **Always validate against direct classical computation before scaling up.**
   Use `qml.matrix(circuit, wire_order=...)` on a tiny case (4-16 basis states)
   and compare the top-left `N x N` block against the expected `A / alpha`
   (where `alpha = sum(|c_i|)`) with `np.allclose`. Do this for the plain block
   encoding AND after adding QSVT phase rotations, before running anything at
   scale ??? LCU sign/normalization bugs are easy to make and hard to debug at
   scale.

4. **QSVT phase count.** For a degree-`d` polynomial, `qml.poly_to_angles(coeffs,
   "QSVT")` returns `d+1` angles. The circuit alternates
   `PCPhase, U, PCPhase, U^dagger, PCPhase, U, ...` ??? for `d=1` (2 angles) this
   is just `PCPhase(angles[0]), U, PCPhase(angles[1])` with a **single**
   application of `U` and no `U^dagger` (a common off-by-one mistake).

5. **Run the sparse circuit as its native gate sequence, not fused into one
   matrix.** For the dense case it's fine (and typically necessary) to fuse the
   whole QSVT circuit into one `qml.QubitUnitary` and run that as a single gate
   on the GPU. For a sparse/structured circuit, fusing defeats the entire point
   ??? always execute it as its actual gate sequence via a `qml.qnode` that
   simply calls the circuit functions.

## Measuring gate count / depth fairly

```python
with qml.tape.QuantumTape() as tape:
    full_qsvt_circuit()
    qml.state()
expanded = qml.tape.QuantumScript(tape.expand(depth=10).operations, tape.measurements)
gate_count = len(expanded.operations)
circuit_depth = expanded.graph.get_depth()
```
Note: `depth=10` expansion may leave some composite dense-matrix operations
(like `BlockEncode`) incompletely decomposed, understating their true
elementary-gate cost, while an explicit gate-based construction (like the LCU
approach above) decomposes fully within the same depth limit. Don't compare
"gate count" between a generic-BlockEncode circuit and an explicit-gate circuit
without this caveat ??? compare wall-clock build time and GPU execution time
instead, which are more trustworthy signals.

## Expected scaling behavior (sanity check)

For a genuinely sparsity-aware LCU encoding on a banded matrix, build time
should grow only with `log2(N)` (i.e., stay nearly flat across a 10-50x
increase in N) and GPU execution time should stay roughly constant, since it's
dominated by fixed per-gate overhead. If you see build time growing
super-linearly with N, you're probably still going through a generic dense
decomposition path somewhere.

