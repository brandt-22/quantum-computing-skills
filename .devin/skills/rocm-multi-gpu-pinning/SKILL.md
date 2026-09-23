---
name: rocm-multi-gpu-pinning
description: Correctly pin a containerized workload to a specific physical AMD GPU on a multi-GPU ROCm host, avoiding a HIP/ROCR visibility collision
allowed-tools:
  - read
  - exec
triggers:
  - user
  - model
---

# Pinning workloads to specific AMD GPUs on a multi-GPU ROCm host

## The problem

On a multi-GPU AMD ROCm host, you generally want to run N independent
containerized workloads, one per physical GPU (e.g. 8 workloads on an 8-GPU
box). The natural approach ??? set `HIP_VISIBLE_DEVICES=<index>` per container ???
often fails in a confusing way:

- Setting `HIP_VISIBLE_DEVICES` (or `ROCR_VISIBLE_DEVICES`) to a **single
  non-zero index** frequently fails outright with `No supported AMD GPU
  devices found` / `Unable to detect any GPU devices`, even though that GPU
  index is valid and idle. Only index `0` reliably works when set alone.
- Restricting container device visibility via `--device=/dev/dri/renderD<N>`
  (exposing only one specific render node instead of `/dev/dri` wholesale)
  does not fix this either ??? only the first render node in the list tends to
  work; others fail the same way. This is because `/dev/kfd` (the actual
  compute device node used by ROCm's HSA runtime) is a single shared node
  covering *all* GPUs ??? restricting `/dev/dri` render nodes doesn't restrict
  KFD-level compute-device enumeration.
- Calling `hipSetDevice(N)` via ctypes *before* PennyLane/Kokkos initializes
  does *not* reliably steer where the workload actually runs either ??? the
  underlying runtime may re-select device 0 regardless, silently executing on
  the wrong (possibly heavily-loaded) GPU. Symptoms: the job takes far longer
  than expected and the intended target GPU's memory usage never increases.

## The fix

**Set only `ROCR_VISIBLE_DEVICES`, and never set `HIP_VISIBLE_DEVICES` at the
same time.** Setting both simultaneously to the same value/list causes a
collision: GPU indices 0 and 1 both resolve to physical GPU 0, and the last
GPU in the visible list becomes completely unreachable (e.g. on an 8-GPU host,
physical GPU 7 was unreachable via any combination of `HIP_VISIBLE_DEVICES`
values 0-7, but became reachable the moment `HIP_VISIBLE_DEVICES` was
dropped and only `ROCR_VISIBLE_DEVICES` was set).

With **only** `ROCR_VISIBLE_DEVICES=<N>` set (single value, no need for a
reordered list), device index N maps cleanly and correctly to physical GPU N ???
verified as a clean bijective identity mapping across all 8 GPUs on an
MI355X host.

```bash
# WRONG ??? do not set both:
docker run -e HIP_VISIBLE_DEVICES=3 -e ROCR_VISIBLE_DEVICES=3 ...

# RIGHT:
docker run -e ROCR_VISIBLE_DEVICES=3 \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  ...
```

## How to verify it's actually working (don't just trust the return code)

Device *creation* succeeding is not proof the workload is running on the
intended GPU ??? it can silently fall back. Verify with a real memory check:

```bash
# baseline
rocm-smi --showmeminfo vram | grep -A1 "GPU\[N\]"
# launch your workload with ROCR_VISIBLE_DEVICES=N, let it run something
# memory-intensive for a few seconds, then check again ??? VRAM usage on GPU N
# should visibly increase, and ONLY on GPU N.
rocm-smi --showmeminfo vram | grep -A1 "GPU\[N\]"
```

If GPU N's memory doesn't move, the workload is not actually running there.

## Practical implication for parallel multi-GPU sweeps

Running 8 independent workloads in parallel (one per GPU via
`ROCR_VISIBLE_DEVICES=0..7`) genuinely parallelizes GPU execution ??? but if the
workload also has a significant CPU-bound phase (e.g. classical matrix
preprocessing before a GPU-accelerated step), that phase still contends for
the host's shared CPU cores across all 8 containers simultaneously. Set
per-container thread limits (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS`) to `total_cores / num_parallel_jobs` to avoid severe
oversubscription (observed: an unthrottled 8-way parallel run caused a ~40x
slowdown per job vs. running alone; properly thread-limited, the same run was
only ~2-3.5x slower per job ??? the difference between useless and a genuine
~2-3x net throughput win).

Also check whether all target GPUs are actually idle before assuming a clean
comparison ??? on a shared host, some GPUs may already be heavily used by other
tenants, causing out-of-memory failures or contention that has nothing to do
with your own workload's scaling behavior.

