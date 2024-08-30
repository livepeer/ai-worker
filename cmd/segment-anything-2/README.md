To run the segment-anything-2 pipeline, the existing Dockerfile is tagged 'base' ( e.g. : FROM livepeer/ai-runner:base ) and the pipeline specific container can be built from cmd/segment-anything-2/Dockerfile.segment_anything_2.

example:
```
cd ai-worker/runner
docker build -t livepeer/ai-runner:base .
docker build -f ../cmd/segment-anything-2/Dockerfile.segment_anything_2 -t livepeer/ai-runner:segment-anything-2 .

docker run --name sam2-runnner -e MODEL_DIR=/models -e PIPELINE=segment-anything-2 -e MODEL_ID=facebook/sam2-hiera-large -e HUGGINGFACE_TOKEN={token} --gpus 0 -p 8002:8000 -v ~/.lpData/models:/models livepeer/ai-runner:segment-anything-2
```

During development, one must similarly specify the ai-runner:segment-anything-2 image:tag combination in lieu of ai-runner:latest