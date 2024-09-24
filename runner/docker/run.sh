#!/bin/bash


ls -lha /usr/lib/x86_64-linux-gnu | grep nvcuv || echo "no nvcuvid found"
echo " ---- JOSH 1 ----"
ls -lha /usr/lib/x86_64-linux-gnu
echo " ---- JOSH 2 ----"
#ls -lha /usr/local/cuda-12.5/targets/x86_64-linux/lib

nvidia-smi

python frames.py &

./mediamtx

kill -SIGINT `pgrep -f frames.py`
