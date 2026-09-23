---
name: openfoam-matrix-extraction
description: Extract the real sparse linear-system (LDU) matrix from an OpenFOAM CFD solve, for use in external linear-algebra/quantum-computing tools
argument-hint: "[OpenFOAM case name, field to extract]"
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

# Extracting a real CFD linear-system matrix from OpenFOAM

OpenFOAM assembles its linear systems internally as an `lduMatrix` (a sparse
representation: diagonal coefficients, lower/upper off-diagonal coefficients,
and owner/neighbour addressing arrays defining mesh face connectivity) ??? it
does not write this matrix out anywhere by default. To get the actual matrix
a real CFD case produces (e.g. for benchmarking a linear solver, or feeding a
real sparse structure into a quantum-linear-algebra experiment), you need to
either use a third-party export utility or patch a solver to dump it directly.
The direct-patch approach below was more reliable in practice than the
`openfoam-matrix-export` third-party library, which has a `fast_matrix_market`
git submodule dependency (with its own nested `dragonbox` dependency) that is
awkward to resolve inside a minimal container without `git` installed.

## Setup

Use an official OpenFOAM Docker image with build tools (a `-run` or default
flavor that includes `wmake`, not just a `-slim` runtime-only image):
```bash
docker pull opencfd/openfoam-default:2312
docker run -d --name ofoam -v /host/workdir:/home/openfoam/work opencfd/openfoam-default:2312 tail -f /dev/null
```

## Choosing a case and mesh size

Copy a standard tutorial case (e.g. the lid-driven cavity,
`$FOAM_TUTORIALS/incompressible/icoFoam/cavity/cavity`) and edit
`system/blockMeshDict` to size the mesh to the number of unknowns you want ???
a `side x side` structured mesh gives exactly `side^2` cells/unknowns (e.g.
64x64 = 4096). Run `blockMesh` then `checkMesh` to confirm the cell count.

## Patching a solver to dump the matrix

Copy the solver source (e.g. `icoFoam.C`) and insert a dump right before the
linear solve ??? for `icoFoam`'s pressure equation, right after
`pEqn.setReference(pRefCell, pRefValue);` and before `pEqn.solve(...)`:

```cpp
{
    OFstream diagF("matrix_diag.txt");
    for (const scalar& v : pEqn.diag()) { diagF << v << nl; }
    OFstream lowerF("matrix_lower.txt");
    for (const scalar& v : pEqn.lower()) { lowerF << v << nl; }
    OFstream upperF("matrix_upper.txt");
    for (const scalar& v : pEqn.upper()) { upperF << v << nl; }
    OFstream srcF("matrix_source.txt");
    for (const scalar& v : pEqn.source()) { srcF << v << nl; }
    OFstream lowAddrF("matrix_lowerAddr.txt");
    for (const label& v : pEqn.lduAddr().lowerAddr()) { lowAddrF << v << nl; }
    OFstream upAddrF("matrix_upperAddr.txt");
    for (const label& v : pEqn.lduAddr().upperAddr()) { upAddrF << v << nl; }
    return 0;  // exit immediately -- one matrix snapshot is usually enough
}
```

Rename the `.C` file, update `Make/files` (`EXE = $(FOAM_USER_APPBIN)/<newname>`),
and build with `wmake` inside the container (`source
/usr/lib/openfoam/openfoam<ver>/etc/bashrc` first). If editing files from
outside the container (e.g. via a mounted volume), note the files will likely
be owned by the container's internal user (often `root`) ??? edit them via
`docker exec` rather than directly from the host to avoid permission errors.

Run the patched solver (`blockMesh` then your new executable) ??? it will exit
immediately after writing the six `matrix_*.txt` files.

## Reconstructing the matrix in Python

OpenFOAM's LDU convention: for internal face `f`, `owner = lowerAddr[f]`,
`neighbour = upperAddr[f]`. Assemble a SciPy sparse matrix as:

```python
import numpy as np, scipy.sparse as sp
diag = np.loadtxt("matrix_diag.txt")
lower = np.loadtxt("matrix_lower.txt")
upper = np.loadtxt("matrix_upper.txt")
lowerAddr = np.loadtxt("matrix_lowerAddr.txt", dtype=int)
upperAddr = np.loadtxt("matrix_upperAddr.txt", dtype=int)
N = len(diag)
rows = np.concatenate([np.arange(N), lowerAddr, upperAddr])
cols = np.concatenate([np.arange(N), upperAddr, lowerAddr])
vals = np.concatenate([diag, upper, lower])
A = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
```

## Sanity-checking the result

For a structured `side x side` 2D mesh with a standard 5-point-stencil
discretization (e.g. `fvm::laplacian`), you should see:
- `bandwidth == side` (max `|i-j|` over nonzero entries) ??? this is the
  signature of row-major flattening of a 2D grid with nearest-neighbor
  coupling.
- 3-5 nonzeros per row (interior cells: 5 = diagonal + 4 neighbors; edge
  cells: 4; corner cells: 3).
- Symmetric, for a pure diffusion/Laplacian-type equation.

```python
coo = A.tocoo()
bandwidth = np.max(np.abs(coo.row - coo.col))
nnz_per_row = np.diff(A.indptr)
```

If these don't match expectations for your mesh, double-check which field's
equation you dumped (e.g. the pressure-correction equation vs. momentum
equation have different structures) and whether the mesh is actually
structured/uniform.

