package main

import (
	"ai-worker/executor/cog"
	"ai-worker/worker"
	"context"
	"log/slog"
	"os"
	"strconv"
	"time"
)

func main() {
	args := os.Args[1:]
	containerImageID := args[0]
	image := args[1]

	gpus := "all"
	outputDir := "output"
	newExecutorFn := func(config worker.ExecutorConfig) worker.Executor {
		return cog.NewExecutor(cog.Config{
			ContainerImageID: config.ContainerImageID,
			GPUs:             config.GPUs,
		})
	}

	w := worker.NewWorker(gpus, outputDir, newExecutorFn)

	slog.Info("Warming container")

	ctx, cancel := context.WithCancel(context.Background())
	if err := w.Warm(ctx, containerImageID); err != nil {
		slog.Error("Error warming container", slog.String("error", err.Error()))
		return
	}

	slog.Info("Warm container is up")

	params := worker.GenerateVideoParams{
		Image:            image,
		MotionBucketID:   127.0,
		NoiseAugStrength: 0.02,
	}

	for i := 0; i < 1; i++ {
		slog.Info("Generating video", slog.Int("num", i))

		jobID := strconv.Itoa(i)
		req := worker.GenerateVideoRequest{
			Params:           params,
			JobID:            jobID,
			ContainerImageID: containerImageID,
		}

		outputPath, err := w.GenerateVideo(ctx, req)
		if err != nil {
			slog.Error("Error generating video", slog.String("error", err.Error()))
			return
		}
		slog.Info("Output written", slog.String("outputPath", outputPath))
	}

	slog.Info("Sleeping 2 seconds and then stopping container")

	time.Sleep(2 * time.Second)

	cancel()

	time.Sleep(1 * time.Second)
}
