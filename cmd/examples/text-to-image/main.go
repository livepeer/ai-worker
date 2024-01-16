package main

import (
	"context"
	"log/slog"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"time"

	"github.com/livepeer/ai-worker/worker"
)

func main() {
	containerName := "text-to-image"
	baseOutputPath := "output"

	containerImageID := "runner"
	gpus := "all"

	modelDir, err := filepath.Abs("models")
	if err != nil {
		slog.Error("Error getting absolute path for modelDir", slog.String("error", err.Error()))
		return
	}

	modelID := "stabilityai/sd-turbo"

	w, err := worker.NewWorker(containerImageID, gpus, modelDir)
	if err != nil {
		slog.Error("Error creating worker", slog.String("error", err.Error()))
		return
	}

	slog.Info("Warming container")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := w.Warm(ctx, containerName, modelID); err != nil {
		slog.Error("Error warming container", slog.String("error", err.Error()))
		return
	}

	slog.Info("Warm container is up")

	args := os.Args[1:]
	runs, err := strconv.Atoi(args[0])
	if err != nil {
		slog.Error("Invalid runs arg", slog.String("error", err.Error()))
		return
	}

	prompt := args[1]

	req := worker.TextToImageJSONRequestBody{
		ModelId: &modelID,
		Prompt:  prompt,
	}

	for i := 0; i < runs; i++ {
		slog.Info("Running text-to-image", slog.Int("num", i))

		urls, err := w.TextToImage(ctx, req)
		if err != nil {
			slog.Error("Error running text-to-image", slog.String("error", err.Error()))
			return
		}

		for j, url := range urls {
			outputPath := path.Join(baseOutputPath, strconv.Itoa(i)+"_"+strconv.Itoa(j)+".png")
			if err := worker.SaveImageB64DataUrl(url, outputPath); err != nil {
				slog.Error("Error saving b64 data url as image", slog.String("error", err.Error()))
				return
			}

			slog.Info("Output written", slog.String("outputPath", outputPath))
		}
	}

	slog.Info("Sleeping 2 seconds and then stopping container")

	time.Sleep(2 * time.Second)

	w.Stop(ctx, containerName)

	time.Sleep(1 * time.Second)
}
