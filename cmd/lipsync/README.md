# Setup for lipsync pipeline

Real3DPortrait https://github.com/yerfor/Real3DPortrait is utilized for the lipsync pipeline

Conda environment is built and configured on the host at /models/models--yerfor--Real3DPortrait/anaconda3 to mimic the paths that will be encountered by the containerized app.

### Setup

On the host machine, run :


``` sudo ./setup_real3dportrait_env.sh ~```

Models are expected to be stored at ~/.lpData/models/ on the host machine, and the final step of this script copies the new conda environment to ~/.lpData/models/models/models--yerfor--Real3DPortrait/anaconda3.

### Usage

```
git clone https://github.com/livepeer/ai-worker.git
cd ai-worker/
docker build -f ./cmd/lipsync/Dockerfile.lipsync -t livepeer/ai-runner:lipsync .
docker run --name lipsync-public -e PIPELINE=lipsync -e MODEL_ID=real3dportrait -e HUGGINGFACE_TOKEN={your token} --gpus 0 -p 8000:8000 -v ~/.lpData/models:/models livepeer/ai-runner:lipsync
```

### Implementation details

The lipsync pipeline in ai-worker/runner invokes a helper script ( cmd/lipsync/run_real3dportrait.sh ) to activate the conda environment inside a shell running in the container, Then, the real3dportrait python inference script is invoked with the parameters provided to the helper script.

### Debugging the lipsync pipeline

Launch.json for Visual Studio Code: 
```
{
    "version": "0.2.0",
    "configurations": [
        
        {
            "name": "Python: Uvicorn",
            "type": "debugpy",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "app.main:app",
                "--log-config", "app/cfg/uvicorn_logging_config.json",
                "--host", "0.0.0.0",
                "--port", "8000"
            ],
            "env": {
                "PIPELINE": "lipsync",
                "MODEL_ID": "real3dportrait",
                "HUGGINGFACE_TOKEN": "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"
            },
            "python": "python",
            "console": "integratedTerminal"
        }
    ]
}

```

Changes to reflect use of the lipsync debug dockerfile need to be made to .devcontainer:

```
diff --git a/runner/.devcontainer/devcontainer.json b/runner/.devcontainer/devcontainer.json
index 3253288..f5c94d6 100644
--- a/runner/.devcontainer/devcontainer.json
+++ b/runner/.devcontainer/devcontainer.json
@@ -5,7 +5,11 @@

        // Image to use for the dev container. More info: https://containers.dev/guide/dockerfile.
        "build": {
-               "dockerfile": "../Dockerfile"
+               "context": "../..",
+               "dockerfile": "../../cmd/lipsync/Dockerfile.debug",
```