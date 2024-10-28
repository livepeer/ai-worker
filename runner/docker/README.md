# Runner Docker Images

This folder contains Dockerfiles for pipelines supported by the Livepeer AI network. The list is maintained by the Livepeer community and audited by the [Core AI team](https://explorer.livepeer.org/treasury/42084921863832634370966409987770520882792921083596034115019946998721416745190). In the future, we will enable custom pipelines to be used with the Livepeer AI network.

## Building a Pipeline-Specific Container

> [!NOTE]
> We are transitioning our existing pipelines to this new structure. As a result, the base container is currently somewhat bloated. In the future, the base image will contain only the necessary dependencies to run any pipeline.

All pipeline-specific containers are built on top of the base container found in the main [runner](../) folder and on [Docker Hub](https://hub.docker.com/r/livepeer/ai-runner). The base container includes the minimum dependencies to run any pipeline, while pipeline-specific containers add the necessary dependencies for their respective pipelines. This structure allows for faster build times, less dependency bloat, and easier maintenance.

### Steps to Build a Pipeline-Specific Container

To build a pipeline-specific container, you need to build the base container first. The base container is tagged as `base`, and the pipeline-specific container is built from the Dockerfile in the pipeline-specific folder. For example, to build the `segment-anything-2` pipeline-specific container, follow these steps:

1. **Navigate to the `ai-worker/runner` Directory**:

   ```bash
   cd ai-worker/runner
    ```

2. **Build the Base Container**:

   ```bash
   docker build -t livepeer/ai-runner:base .
   ```

   This command builds the base container and tags it as `livepeer/ai-runner:base`.

3. **Build the `segment-anything-2` Pipeline-Specific Container**:

   ```bash
   docker build -f docker/Dockerfile.segment_anything_2 -t livepeer/ai-runner:segment-anything-2 .
   ```

   This command builds the `segment-anything-2` pipeline-specific container using the Dockerfile located at [docker/Dockerfile.segment_anything_2](docker/Dockerfile.segment_anything_2) and tags it as `livepeer/ai-runner:segment-anything-2`.

### Steps to Build a Realtime Video AI Container

   ```bash
   docker build -t livepeer/ai-runner:live-base . -f docker/Dockerfile.live-base
   docker build -t livepeer/ai-runner:live-multimedia -f docker/Dockerfile.live-multimedia .
   docker build -t livepeer/ai-runner:live-stream-diffusion -f docker/Dockerfile.live-stream-diffusion .
   docker build -t livepeer/ai-runner:live-apps -f docker/Dockerfile.live-apps .
   ```

   Then, you can run and test the Live Container with the following commands:
   ```bash
   docker run -it --rm --name video-to-video -e PIPELINE=live-video-to-video -e MODEL_ID=KBlueLeaf/kohaku-v2.1 --gpus all -p 8000:8000 -v ./models:/models livepeer/ai-runner:live-apps

   curl --location -H "Content-Type: application/json" 'http://localhost:8000/live-video-to-video' -X POST -d '{"stream_url":"http://<url-to-trickle-pull>"}'
   ```