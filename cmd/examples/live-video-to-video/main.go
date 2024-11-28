package main

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"log/slog"
	"os"
	"path/filepath"
	"image"
	_ "image/jpeg"
    _ "image/png"
	"sync"
	"time"
	"fmt"

	"github.com/pebbe/zmq4"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/filters"
	docker "github.com/docker/docker/client"
	"github.com/livepeer/ai-worker/worker"
)

func sendImages(ctx context.Context, imagePath string, fps int) error {
    publisher, err := zmq4.NewSocket(zmq4.PUB)
    if err != nil {
        return fmt.Errorf("failed to create ZMQ PUB socket: %v", err)
    }
    defer publisher.Close()

    sendAddress := "tcp://*:5555"
    err = publisher.Bind(sendAddress)
    if err != nil {
        return fmt.Errorf("failed to bind ZMQ PUB socket: %v", err)
    }

    imageBytes, err := os.ReadFile(imagePath)
    if err != nil {
        return fmt.Errorf("failed to read image file: %v", err)
    }

    interval := time.Duration(1e9 / fps)

    slog.Info(fmt.Sprintf("Sending images at %d FPS to %s", fps, sendAddress))

    ticker := time.NewTicker(interval)
    defer ticker.Stop()

    for {
        select {
        case <-ctx.Done():
            return nil
        case <-ticker.C:
            _, err = publisher.SendBytes(imageBytes, 0)
            if err != nil {
                slog.Error("Failed to send image bytes", slog.String("error", err.Error()))
            }
        }
    }
}


func receiveImages(ctx context.Context) error {
    time.Sleep(5 * time.Second)

    subscriber, err := zmq4.NewSocket(zmq4.SUB)
    if err != nil {
        return fmt.Errorf("failed to create ZMQ SUB socket: %v", err)
    }
    defer subscriber.Close()

    receiveAddress := "tcp://127.0.0.1:5556"
    err = subscriber.Connect(receiveAddress)
    if err != nil {
        return fmt.Errorf("failed to connect ZMQ SUB socket: %v", err)
    }

    err = subscriber.SetSubscribe("")
    if err != nil {
        return fmt.Errorf("failed to subscribe to all messages: %v", err)
    }

    slog.Info(fmt.Sprintf("Receiving images on %s", receiveAddress))

    startTime := time.Now()
    numImages := 0

    for {
        select {
        case <-ctx.Done():
            return nil
        default:
            imageBytes, err := subscriber.RecvBytes(0)
            if err != nil {
                slog.Error("Failed to receive image bytes", slog.String("error", err.Error()))
                continue
            }

            reader := bytes.NewReader(imageBytes)
            _, _, err = image.Decode(reader)
            if err != nil {
                slog.Error("Failed to decode received image", slog.String("error", err.Error()))
                continue
            }

            numImages++

            currentTime := time.Now()
            elapsedTime := currentTime.Sub(startTime)
            if elapsedTime >= time.Second {
                fps := float64(numImages) / elapsedTime.Seconds()
                slog.Info(fmt.Sprintf("Receiving FPS: %.2f", fps))
                startTime = currentTime
                numImages = 0
            }
        }
    }
}

func main() {
    aiModelsDir := flag.String("aiModelsDir", "runner/models", "path to the models directory")
    fps := flag.Int("fps", 5, "Frames per second to send")
    modelID := flag.String("modelid", "noop", "Model ID for the live pipeline")
	imagePath := flag.String("imagePath", "/home/user/ai-worker/runner/example_data/image.jpeg", "Path to the image to send")
    flag.Parse()

    pipeline := "live-video-to-video"
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

    // Create a context with a 5-minute timeout
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
    defer cancel()

	// Remove existing containers if any
    existingContainers, err := dockerClient.ContainerList(ctx, container.ListOptions{
        Filters: filters.NewArgs(
            filters.Arg("label", "creator=ai-worker"),
            filters.Arg("name", pipeline+"_"+*modelID),
        ),
        All: true, // include stopped containers
    })
    if err != nil {
        slog.Error("Error listing existing containers", slog.String("error", err.Error()))
        return
    }
    for _, _container := range existingContainers {
        slog.Info("Removing existing container", slog.String("container_id", _container.ID))
        err := dockerClient.ContainerRemove(ctx, _container.ID, container.RemoveOptions{
            Force: true,
        })
        if err != nil {
            slog.Error("Error removing container", slog.String("container_id", _container.ID), slog.String("error", err.Error()))
            return
        }
    }

    slog.Info("Warming container")

    if err := w.Warm(ctx, pipeline, *modelID, worker.RunnerEndpoint{}, worker.OptimizationFlags{}); err != nil {
        slog.Error("Error warming container", slog.String("error", err.Error()))
        return
    }

    slog.Info("Warm container is up")

    streamProtocol := "zeromq"
    req := worker.GenLiveVideoToVideoJSONRequestBody{
        ModelId:      modelID,
        SubscribeUrl: "tcp://172.17.0.1:5555",
        PublishUrl:   "tcp://*:5556",
        StreamProtocol: &streamProtocol,
    }

    slog.Info("Running live-video-to-video")

    resp, err := w.LiveVideoToVideo(ctx, req)
    if err != nil {
        slog.Error("Error running live-video-to-video", slog.String("error", err.Error()))
        return
    }

    // The response will be empty since there's no input stream
    slog.Info("Got response", slog.Any("response", resp))

    var wg sync.WaitGroup

    // Start sending images after 2 seconds
    wg.Add(1)
    go func() {
        defer wg.Done()
        select {
        case <-time.After(2 * time.Second):
            err := sendImages(ctx, *imagePath, *fps)
            if err != nil {
                slog.Error("Error in sendImages", slog.String("error", err.Error()))
            }
        case <-ctx.Done():
            return
        }
    }()

    // Start receiving images after 5 seconds (3 seconds after sending starts)
    wg.Add(1)
    go func() {
        defer wg.Done()
        select {
        case <-time.After(5 * time.Second):
            err := receiveImages(ctx)
            if err != nil {
                slog.Error("Error in receiveImages", slog.String("error", err.Error()))
            }
        case <-ctx.Done():
            return
        }
    }()

    // Wait for either the context to be done or the goroutines to finish
    done := make(chan struct{})
    go func() {
        wg.Wait()
        close(done)
    }()

    select {
    case <-ctx.Done():
        slog.Info("Context done, stopping")
    case <-done:
        slog.Info("All goroutines finished")
    }

    w.Stop(ctx)
}
