# ai-worker

Note: This repo contains prototype code that is changing quickly and is not intended for production use at this point.

## Build

Follow the README instructions in the [runner](https://github.com/livepeer/ai-worker/tree/main/runner) directory to download model checkpoints and build the runner image.

Generate Go code for the runner OpenAPI spec:

```
make codegen
```

## Run examples

```
go run cmd/examples/text-to-image/main.go <RUNS> <PROMPT>
```