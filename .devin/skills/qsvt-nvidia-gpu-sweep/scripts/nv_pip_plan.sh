IMG=mlperf-nvidia:llama2_70b_lora-pyt
docker run --rm --name qsvt_b300_pipplan --memory 64g --cpus 8 --entrypoint bash $IMG -c '
pip --version
pip install --dry-run --quiet --report /tmp/r.json pennylane pennylane-lightning pennylane-lightning-gpu pyqsp pytest 2>&1 | tail -n 5
python3 - <<PY
import json
r = json.load(open("/tmp/r.json"))
items = sorted((i["metadata"]["name"], i["metadata"]["version"]) for i in r["install"])
print("WOULD INSTALL (already-present packages are not listed):")
for n, v in items: print(f"  {n}=={v}")
PY
' 2>&1 | tail -n 60
echo END
