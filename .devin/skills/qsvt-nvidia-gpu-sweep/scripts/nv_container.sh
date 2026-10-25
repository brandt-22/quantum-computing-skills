# Helper: run a command inside the isolated work container (one GPU, bounded CPU/RAM, libs mounted from the persistent dir).
# usage: bash nv_container.sh <gpu index> <name> <command...>
GPU=$1; NAME=$2; shift 2
L=/libs
docker run --rm --name "$NAME" --gpus "device=$GPU" --memory 512g --cpus 24 --pids-limit 4096 \
  -v /root/qsvt_b300/work:/work -v /root/qsvt_b300/libs:/libs \
  -e PYTHONPATH=/libs -e OMP_NUM_THREADS=8 -e PYTHONUNBUFFERED=1 \
  -e LD_LIBRARY_PATH=$L/nvidia/cuda_runtime/lib:$L/nvidia/cublas/lib:$L/nvidia/cusparse/lib:$L/nvidia/nvjitlink/lib:$L/nvidia/cuda_nvrtc/lib:$L/cuquantum/lib:/usr/local/cuda/compat/lib \
  -w /work --entrypoint bash mlperf-nvidia:llama2_70b_lora-pyt -c "$*"
