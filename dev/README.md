# Development Guide

This guide provides instructions for setting up the development environment and debugging the [AI worker](https://github.com/livepeer/ai-worker) repository.

## Debugging

### Local Debugging

To directly debug the AI worker, use the go scripts in the [cmd/examples](https://github.com/livepeer/go-livepeer/tree/ai-video/cmd) folder to test and debug the AI worker. Run these scripts with [Golang](https://go.dev/) or use [Vscode](https://code.visualstudio.com/) with the [golang extension](https://code.visualstudio.com/docs/languages/go) for debugging. Future updates will include tests to enhance the development pipeline.

### Debugging within go-livepeer

To debug the AI worker within the context of the [go-livepeer](https://github.com/livepeer/go-livepeer/tree/ai-video) project, replace the go module reference in the go.mod file of the go-livepeer project with the path to your local AI worker repository:

```bash
go mod edit -replace github.com/livepeer/ai-worker=../path/to/ai-worker
```

This setup allows you to debug the AI worker package directly when running the [go-livepeer](https://github.com/livepeer/go-livepeer/tree/ai-video) software.
