# Development Documentation

This guide aims to assist developers working on the [AI runner](https://github.com/livepeer/ai-worker/tree/main/runner), offering detailed instructions for debugging and setting up the development environment. For general information about the AI runner, refer to the [AI Runner README](../README.md).

## Debugging

### Using the Provided DevContainer

Leverage the [VSCode DevContainer](https://code.visualstudio.com/docs/remote/containers) for an efficient debugging experience with the [AI runner](https://github.com/livepeer/ai-worker/tree/main/runner). This configuration automatically prepares a development environment equipped with all necessary tools and dependencies.

**Quickstart with DevContainer:**

1. **Install** [VSCode](https://code.visualstudio.com/download) and the [Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).
2. **Clone** the repository and open it in VSCode.
3. **Navigate** to the `runner` directory.
4. **Open in Container**: Click "Reopen in Container" when prompted, or manually initiate it by pressing `F1`, typing "Reopen in Container", and pressing `Enter`.
5. **Initialization**: The initial build may take a few minutes. Subsequent starts are faster.
6. **Begin Debugging**: The AI runner is now set up for debugging.

For more, see the [VSCode documentation on DevContainers](https://code.visualstudio.com/docs/devcontainers/containers).

### Debugging with External Services

To debug the AI runner when it operates within a container orchestrated by external services, such as [go-livepeer](https://github.com/livepeer/go-livepeer/tree/ai-video), follow these steps to attach to the container:

1. **Directory**: Ensure you're in the `runner` directory.
2. **Build the AI Runner Image**:

   ```bash
   docker build -t livepeer/ai-runner:base .
   ```

3. **Build the Debug Image**:

   ```bash
   docker build -f ./dev/Dockerfile.debug -t livepeer/ai-runner:latest .
   ```

4. **Apply the Debug Patch**: Implement the required code modifications to facilitate debugger attachment and expose the necessary ports.

   ```bash
   cd .. && git apply ./runner/dev/patches/debug.patch && cd runner
   ```

5. **Attach and Debug**: Follow the [guidance on attaching to a running container](https://code.visualstudio.com/docs/python/debugging#_command-line-debugging) for details. To attach to the AI runner, use the following configuration:

   ```json
   {
     "version": "0.2.0",
     "configurations": [
       {
         "name": "Python Debugger: Remote Attach",
         "type": "debugpy",
         "request": "attach",
         "connect": {
           "host": "0.0.0.0",
           "port": 5678
         },
         "pathMappings": [
           {
             "localRoot": "${workspaceFolder}",
             "remoteRoot": "."
           }
         ]
       }
     ]
   }
   ```

6. **Revert Changes**: After debugging, undo the debug patch.

   ```bash
   cd .. && git apply -R ./runner/dev/patches/debug.patch && cd runner
   ```

7. **Rebuild the Image**: Execute a rebuild of the image, ensuring to exclude the debug changes.

   ```bash
   docker build -t livepeer/ai-runner:latest .
   ```
