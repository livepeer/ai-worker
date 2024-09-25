#!/bin/bash

# NB: terminating the container does NOT
# terminate mediamtx! need to propagate
# the signal for a clean shutdown

# put python into the background
python frames2video.py &

# blocks until exit
./mediamtx

# clean python
kill -SIGINT `pgrep -f frames.py`
