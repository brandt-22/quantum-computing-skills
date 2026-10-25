import os, time, subprocess, sys
import numpy as np
import pennylane as qml
print("pennylane", qml.__version__, "| numpy", np.__version__, "| python", sys.version.split()[0])
print("nvidia-smi inside container:\n" + subprocess.run(["nvidia-smi", "--query-gpu=index,name,compute_cap,memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout)
for name in ("lightning.gpu", "lightning.qubit"):
    n = 24
    try:
        dev = qml.device(name, wires=n)
        @qml.qnode(dev)
        def c():
            for w in range(n): qml.Hadamard(w)
            qml.CNOT([0, 1]); qml.RZ(0.3, 1)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])
        t = time.time(); r = c(); dt = time.time() - t
        print(f"{name}: OK  class={type(dev).__name__}  time={dt:.2f}s  expval={float(r[0]):.6f} probs={np.round(np.asarray(r[1]),4)}")
    except Exception as e:
        print(f"{name}: FAILED {type(e).__name__}: {str(e)[:600]}")
