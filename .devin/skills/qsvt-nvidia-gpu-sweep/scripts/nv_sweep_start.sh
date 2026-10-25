echo "GPU 0 idle check:"; nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "containers running: $(docker ps -q | wc -l)"
rm -f /root/qsvt_b300/work/results/sweep_gpu.json /root/qsvt_b300/work/results/sweep.log
nohup bash /root/qsvt_b300/nv_container.sh 0 qsvt_b300_sweep "python3 -W ignore /work/programs/run_sweep.py --device lightning.gpu --limit 600 --out /work/results/sweep_gpu.json" > /root/qsvt_b300/work/results/sweep.log 2>&1 &
sleep 20
cat /root/qsvt_b300/work/results/sweep.log
echo END
