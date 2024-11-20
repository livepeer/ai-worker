package main

import (
	"context"
	"errors"
	"flag"
	"log/slog"
	"os"
	"path/filepath"
	"time"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/filters"
	docker "github.com/docker/docker/client"
	"github.com/livepeer/ai-worker/worker"
)

func main() {
	aiModelsDir := flag.String("aiModelsDir", "runner/models", "path to the models directory")
	flag.Parse()

	pipeline := "live-video-to-video"
	modelID := "noop" // modelID is used for the name of the live pipeline
	// modelID = "liveportrait"
	// modelID = "streamdiffusion"
	// modelID = "comfyui"
	defaultImageID := "livepeer/ai-runner:latest"
	gpus := []string{"0"}

	modelsDir, err := filepath.Abs(*aiModelsDir)
	if errors.Is(err, os.ErrNotExist) {
		slog.Error("Directory does not exist", slog.String("path", *aiModelsDir))
		return
	} else if err != nil {
		slog.Error("Error getting absolute path for 'aiModelsDir'", slog.String("error", err.Error()))
		return
	}

	w, err := worker.NewWorker(defaultImageID, gpus, modelsDir)
	if err != nil {
		slog.Error("Error creating worker", slog.String("error", err.Error()))
		return
	}

	dockerClient, err := docker.NewClientWithOpts(docker.FromEnv, docker.WithAPIVersionNegotiation())
	if err != nil {
		slog.Error("Error creating docker client", slog.String("error", err.Error()))
		return
	}

	slog.Info("Warming container")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := w.Warm(ctx, pipeline, modelID, worker.RunnerEndpoint{}, worker.OptimizationFlags{}); err != nil {
		slog.Error("Error warming container", slog.String("error", err.Error()))
		return
	}

	slog.Info("Warm container is up")

	req := worker.GenLiveVideoToVideoJSONRequestBody{
		ModelId: &modelID,
	}

	slog.Info("Running live-video-to-video")

	resp, err := w.LiveVideoToVideo(ctx, req)
	if err != nil {
		slog.Error("Error running live-video-to-video", slog.String("error", err.Error()))
		return
	}

	// The response will be empty since there's no input stream
	slog.Info("Got response", slog.Any("response", resp))

	// Check container status every 5 seconds for up to 2 minutes
	timeout := time.After(2 * time.Minute)
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	// Filter for our specific container
	f := filters.NewArgs()
	f.Add("label", "creator=ai-worker")
	f.Add("name", pipeline+"_"+modelID)

	for {
		select {
		case <-timeout:
			slog.Error("Container did not stop within 2 minutes")
			w.Stop(ctx)
			return
		case <-ticker.C:
			containers, err := dockerClient.ContainerList(ctx, container.ListOptions{
				Filters: f,
			})
			if err != nil {
				slog.Error("Error listing containers", slog.String("error", err.Error()))
				w.Stop(ctx)
				return
			}
			if len(containers) == 0 {
				slog.Info("Container stopped as expected")
				return
			}
			slog.Info("Container still running, checking again in 5 seconds",
				slog.Int("container_count", len(containers)),
				slog.String("container_name", containers[0].Names[0]))
		}
	}
}
