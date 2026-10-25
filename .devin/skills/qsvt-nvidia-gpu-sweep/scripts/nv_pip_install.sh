IMG=mlperf-nvidia:llama2_70b_lora-pyt
docker run --rm --name qsvt_b300_pip --memory 64g --cpus 8 -v /root/qsvt_b300/libs:/libs --entrypoint bash $IMG -c '
pip install --no-deps --quiet --target /libs \
  appdirs==1.4.4 autograd==1.8.0 autoray==0.8.4 cachetools==7.2.4 custatevec-cu12==1.15.0 diastatic-malt==2.15.3 iniconfig==2.3.1 \
  nvidia-cublas-cu12==12.9.2.10 nvidia-cuda-nvrtc-cu12==12.9.86 nvidia-cuda-runtime-cu12==12.9.79 nvidia-cusparse-cu12==12.5.10.65 \
  nvidia-nvjitlink-cu12==12.9.86 pennylane==0.45.1 pennylane_lightning==0.45.0 pennylane_lightning_gpu==0.45.0 pluggy==1.7.0 \
  pyqsp==0.2.0 pytest==9.1.1 rustworkx==0.18.1 scipy-openblas32==0.3.34.237.2 termcolor==3.3.0 tomlkit==0.15.2 2>&1 | tail -n 5
echo "pip exit: $?"
'
du -sh /root/qsvt_b300/libs; ls /root/qsvt_b300/libs | head -n 40
echo END
