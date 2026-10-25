echo "GPU 0 idle check:"; nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader
M=case07_n32_nu0.01_U1_Ux_call11
bash /root/qsvt_b300/nv_container.sh 0 qsvt_b300_validate "
for dev in lightning.gpu lightning.qubit; do
  echo \"=========== \$dev ===========\"
  python3 -W ignore /work/programs/qsvt_solve_case.py --matrix /work/matrices/$M --device \$dev --out /work/results/validate_\$dev.json 2>&1 | grep -E 'degree|p_success|rel_err|qsvt_plus|qsvt_minus|total_s|visible_gpus|\"device\"'
done"
echo "containers left: $(docker ps -q | wc -l)"
echo END
