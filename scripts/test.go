package main

import (
	"context"
	"fmt"

	"github.com/livepeer/ai-worker/worker"
)

func main() {
	gpus := []string{"0"}
	AIWorker, err := worker.NewWorker("livepeer/ai-runner:latest", gpus, "/home/ricks/.lpData/models")
	if err != nil {
		fmt.Errorf("Error creating AI worker: %v", err)
		return
	}

	ctx, cancel := context.WithCancel(context.Background())
	Pipeline := "image-to-image"
	ModelID := "ByteDance/SDXL-Lightning"
	URL := ""
	Token := ""
	endpoint := worker.RunnerEndpoint{URL: URL, Token: Token}
	if err := AIWorker.Warm(ctx, Pipeline, ModelID, endpoint); err != nil {
		fmt.Printf("Error creating container: %v\n", err)
		cancel()
		return
	}
	fmt.Println("AI worker started")
}
