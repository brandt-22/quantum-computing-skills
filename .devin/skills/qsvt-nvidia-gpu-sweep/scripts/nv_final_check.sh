echo "containers running: $(docker ps -q | wc -l) | containers (any state) named qsvt: $(docker ps -aq --filter name=qsvt | wc -l)"
echo "GPU state:"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "images count: $(docker images -q | sort -u | wc -l)  (was 13 repo:tag lines at start; nothing pulled)"
echo "my footprint:"; du -sh /root/qsvt_b300 /root/qsvt_b300/libs /root/qsvt_b300/work 2>/dev/null
ls /root/qsvt_b300/work/results | wc -l
echo "host python site-packages untouched check (pennylane on host?):"; python3 -c "import importlib.util as u; print('pennylane on host:', u.find_spec('pennylane') is not None)"
echo END
