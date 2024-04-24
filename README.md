# ai-worker

> [!WARNING]
> This is a prototype repository undergoing rapid changes. It's not intended for production use yet.

This repository hosts the AI worker and runner for processing inference requests on the Livepeer AI subnet.

## Overview

The AI worker repository includes:

- **Runner**: The [AI runner](https://github.com/livepeer/ai-worker/tree/main/runner), a containerized Python application, processes inference requests on Livepeer AI's Pipelines and models, providing a REST API for model interaction.

- **Worker**: The [AI worker](https://github.com/livepeer/ai-worker) allows the [ai-video](https://github.com/livepeer/go-livepeer/tree/ai-video) branch of [go-livepeer](https://github.com/livepeer/go-livepeer/tree/ai-video) to interact with the AI runner. It includes golang API bindings, a worker for routing inference requests, and a Docker manager for AI runner containers.

### Runner

The AI runner's code is in the [runner](https://github.com/livepeer/ai-worker/tree/main/runner) directory. For more details, see the [AI runner README](./runner/README.md).

### Worker

The AI worker's code is in the [worker](https://github.com/livepeer/ai-worker/tree/main/worker) directory. It includes:

- **Golang API Bindings**: Generated from the AI runner's OpenAPI spec using `make codegen`.
- **Worker**: Listens for inference requests from the Livepeer AI subnet and routes them to the AI runner.
- **Docker Manager**: Manages AI runner containers.

## Build

The AI worker and runner are designed to work with the [ai-video](https://github.com/livepeer/go-livepeer/tree/ai-video) branch of [go-livepeer](https://github.com/livepeer/go-livepeer/tree/ai-video). You can run both independently for testing. To build the AI worker locally and run examples, follow these steps:

1. Follow the README instructions in the [runner](./runner/README.md) directory to download model checkpoints and build the runner image.
2. Generate Go bindings for the runner OpenAPI spec with `make codegen`.
3. Run any examples in the `cmd/examples` directory, e.g., `go run cmd/examples/text-to-image/main.go <RUNS> <PROMPT>`.

## Development documentation

For more on developing and debugging the AI runner, see the [development documentation](./dev/README.md).
