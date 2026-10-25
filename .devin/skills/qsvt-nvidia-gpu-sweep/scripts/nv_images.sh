echo "=== all images ==="; docker images --format '{{.Repository}}:{{.Tag}}  {{.Size}}  {{.ID}}' | sed -n '1,60p'
for img in mlperf-nvidia:llama2_70b_lora-pyt mlperf-nvidia:flux1-pyt; do
  echo; echo "######## $img ########"
  docker run --rm --entrypoint python3 $img -c "
import sys, importlib.util as u
print('python', sys.version.split()[0])
for m in ('numpy','scipy','torch','cupy','pennylane','pennylane_lightning','custatevec','cuquantum','pyqsp','pytest','qiskit'):
    s = u.find_spec(m)
    if s:
        try:
            mod = __import__(m); print('  has', m, getattr(mod, '__version__', ''))
        except Exception as e: print('  has', m, '(import error:', str(e)[:60], ')')
    else: print('  no ', m)
import os; print('CUDA_VERSION', os.environ.get('CUDA_VERSION'), '| NVIDIA_VISIBLE_DEVICES', os.environ.get('NVIDIA_VISIBLE_DEVICES'))
" 2>&1 | tail -n 16
done
echo END
