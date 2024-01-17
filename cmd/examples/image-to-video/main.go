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
	"github.com/oapi-codegen/runtime/types"
)

func main() {
	containerName := "image-to-video"
	baseOutputPath := "output"

	containerImageID := "runner"
	gpus := "all"

	modelDir, err := filepath.Abs("runner/models")
	if err != nil {
		slog.Error("Error getting absolute path for modelDir", slog.String("error", err.Error()))
		return
	}

	modelID := "stabilityai/stable-video-diffusion-img2vid-xt"

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

	imagePath := args[1]

	imageBytes, err := os.ReadFile(imagePath)
	if err != nil {
		slog.Error("Error reading image", slog.String("imagePath", imagePath))
		return
	}
	imageFile := types.File{}
	imageFile.InitFromBytes(imageBytes, imagePath)

	req := worker.ImageToVideoMultipartRequestBody{
		Image:   imageFile,
		ModelId: &modelID,
	}

	for i := 0; i < runs; i++ {
		slog.Info("Running image-to-video", slog.Int("num", i))

		resp, err := w.ImageToVideo(ctx, req)
		if err != nil {
			slog.Error("Error running image-to-video", slog.String("error", err.Error()))
			return
		}

		for j, batch := range resp.Frames {
			dirPath := path.Join(baseOutputPath, strconv.Itoa(i)+"_"+strconv.Itoa(j))

			for frameNum, media := range batch {
				if err := os.MkdirAll(dirPath, os.ModePerm); err != nil {
					slog.Error("Error creating dir", slog.String("dir", dirPath))
					return
				}

				outputPath := path.Join(dirPath, strconv.Itoa(frameNum)+".png")
				if err := worker.SaveImageB64DataUrl(media.Url, outputPath); err != nil {
					slog.Error("Error saving b64 data url as image", slog.String("error", err.Error()))
					return
				}
			}
			slog.Info("Outputs written", slog.String("dirPath", dirPath))
		}
	}

	slog.Info("Sleeping 2 seconds and then stopping container")

	time.Sleep(2 * time.Second)

	w.Stop(ctx, containerName)

	time.Sleep(1 * time.Second)
}
