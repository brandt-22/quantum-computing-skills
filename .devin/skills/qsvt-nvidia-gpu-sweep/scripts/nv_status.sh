tail -n 5 /root/qsvt_b300/work/results/sweep.log | cut -c1-230
docker ps --format '{{.Names}} {{.Status}}'
echo "containers running: $(docker ps -q | wc -l)"
echo END
