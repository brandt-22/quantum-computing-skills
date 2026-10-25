echo "=== os / kernel ==="; cat /etc/os-release | head -n 3; uname -r
echo "=== gpus ==="; nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,compute_cap --format=csv
echo "=== gpu processes ==="; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
echo "=== driver/cuda ==="; nvidia-smi | head -n 4 | tail -n 2
echo "=== docker ==="; docker --version; docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
echo "=== docker images ==="; docker images --format '{{.Repository}}:{{.Tag}}  {{.Size}}' | head -n 25
echo "=== runtimes ==="; docker info 2>/dev/null | grep -i -E "runtimes|default runtime|storage driver|docker root"
echo "=== disk ==="; df -h / /root /var/lib/docker 2>/dev/null; echo; df -h | grep -v -E "tmpfs|overlay|udev" | head -n 12
echo "=== memory/cpu ==="; free -g | head -n 2; nproc
echo "=== python / pip on host (read only) ==="; which python3 pip3 2>&1; python3 --version 2>&1
echo "=== existing dirs in /root ==="; ls -la /root | head -n 30
echo "=== other users / logins ==="; who; last -n 5 2>/dev/null | head -n 6
echo "=== internet? ==="; (curl -s -m 8 -o /dev/null -w "pypi: %{http_code}\n" https://pypi.org/simple/pip/ ; curl -s -m 8 -o /dev/null -w "dockerhub: %{http_code}\n" https://registry-1.docker.io/v2/) 2>&1
echo END
