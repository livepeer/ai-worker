#!/bin/bash
set -ex

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_room> <output_room>"
    exit 1
fi

INPUT_ROOM=$1
OUTPUT_ROOM=$2

docker build -t livepeer/ai-runner:live-apps -f docker/Dockerfile.live-apps .
docker run -it --rm --name live-video-to-video \
  -e PIPELINE=live-video-to-video \
  -e MODEL_ID=streamkohaku \
  --gpus all \
  -p 9000:8000 \
  -v ./models:/models \
  livepeer/ai-runner:live-apps 2>&1 | tee ./run-lv2v.log &
DOCKER_PID=$!

# make sure to kill the container when the script exits
trap 'docker rm -f live-video-to-video' EXIT

set +x

echo "Waiting for server to start..."
while ! grep -aq "Uvicorn running" ./run-lv2v.log; do
  sleep 1
done
sleep 2
echo "Starting pipeline from ${INPUT_ROOM} to ${OUTPUT_ROOM}..."

set -x

curl -vvv http://localhost:9000/live-video-to-video/ \
  -H 'Content-Type: application/json' \
  -d "{\"publish_url\":\"https://wwgcyxykwg9dys.transfix.ai/trickle/${OUTPUT_ROOM}\",\"subscribe_url\":\"https://wwgcyxykwg9dys.transfix.ai/trickle/${INPUT_ROOM}\"}"

# let docker container take over
wait $DOCKER_PID
