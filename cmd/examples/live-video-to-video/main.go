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
	"sync"
	"time"
	"fmt"
    "math"
    "sort"

	"github.com/pebbe/zmq4"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/filters"
	docker "github.com/docker/docker/client"
	"github.com/livepeer/ai-worker/worker"
)

func sendImages(ctx context.Context, imagePath string, inputFps int) error {
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

    interval := time.Duration(1e9 / inputFps)

    slog.Info(fmt.Sprintf("Sending images at %d FPS to %s", inputFps, sendAddress))

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

func printFPSStatistics(fpsList []float64, expOutputFps int) {
    if len(fpsList) < 5 {
        slog.Info("Not enough FPS values collected (minimum 5 required)")
        return
    }

    // Remove first 5 sec values (warm up)
    fpsList = fpsList[5:]
    
    if len(fpsList) == 0 {
        slog.Info("No FPS values remaining after removing first 5 values")
        return
    }

    // Sort the list for percentile calculations
    sorted := make([]float64, len(fpsList))
    copy(sorted, fpsList)
    sort.Float64s(sorted)

    // Calculate statistics
    min := sorted[0]
    max := sorted[len(sorted)-1]
    
    var sum float64
    for _, v := range sorted {
        sum += v
    }
    avg := sum / float64(len(sorted))

    // Calculate median (p50)
    median := calculatePercentile(sorted, 50)
    p90 := calculatePercentile(sorted, 90)
    p99 := calculatePercentile(sorted, 99)

    // Print statistics
    slog.Info(fmt.Sprintf("FPS Statistics:"+
        "\nMin: %.2f"+
        "\nMax: %.2f"+
        "\nAvg: %.2f"+
        "\nMedian (P50): %.2f"+
        "\nP90: %.2f"+
        "\nP99: %.2f\n",
        min, max, avg, median, p90, p99))

    if min >= float64(expOutputFps) {
        slog.Info("TEST PASSED!")
    } else {
        slog.Info("TEST FAILED!")
    }
}

func calculatePercentile(sorted []float64, percentile float64) float64 {
    index := (percentile / 100.0) * float64(len(sorted)-1)
    i := int(math.Floor(index))
    fraction := index - float64(i)

    if i+1 >= len(sorted) {
        return sorted[i]
    }

    return sorted[i] + fraction*(sorted[i+1]-sorted[i])
}


func receiveImages(ctx context.Context, expOutputFps int) error {
    time.Sleep(2 * time.Second)

    subscriber, err := zmq4.NewSocket(zmq4.SUB)
    if err != nil {
        return fmt.Errorf("failed to create ZMQ SUB socket: %v", err)
    }
    defer subscriber.Close()

    receiveAddress := "tcp://*:5556"
    err = subscriber.Bind(receiveAddress)
    if err != nil {
        return fmt.Errorf("failed to connect ZMQ SUB socket: %v", err)
    }

    err = subscriber.SetSubscribe("")
    if err != nil {
        return fmt.Errorf("failed to subscribe to all messages: %v", err)
    }

    // Set receive timeout
    err = subscriber.SetRcvtimeo(1 * time.Second)
    if err != nil {
        return fmt.Errorf("failed to set receive timeout: %v", err)
    }

    slog.Info(fmt.Sprintf("Receiving images on %s", receiveAddress))

    startTime := time.Now()
    numImages := 0
    var fpsList []float64

    for {
        select {
        case <-ctx.Done():
            printFPSStatistics(fpsList, expOutputFps)
            return nil
        default:
            imageBytes, err := subscriber.RecvBytes(0)
            if err != nil {
                if errors.Is(err, os.ErrDeadlineExceeded) {
                    slog.Error("1sec timelimit exceeded", slog.String("error", err.Error()))
                    continue
                } else {
                    slog.Error("Failed to receive image bytes", slog.String("error", err.Error()))
                    continue
                }
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
                currentFPS := float64(numImages) / elapsedTime.Seconds()
                fpsList = append(fpsList, currentFPS)
                slog.Info(fmt.Sprintf("Receiving FPS: %.2f", currentFPS))
                startTime = currentTime
                numImages = 0
            }
        }
    }
}

func main() {
    aiModelsDir := flag.String("aimodelsdir", "runner/models", "path to the models directory")
    inputFps := flag.Int("inputfps", 30, "Frames per second to send")
    modelID := flag.String("modelid", "noop", "Model ID for the live pipeline")
	imagePath := flag.String("imagepath", "runner/example_data/image.jpeg", "Path to the image to send")
    expOutputFps := flag.Int("expoutputfps", 27, "Minimum expected output FPS")
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

    // Create a context with a 1.5-minute timeout
    ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
    defer cancel()

	// Remove existing containers if any
    existingContainers, err := dockerClient.ContainerList(ctx, container.ListOptions{
        Filters: filters.NewArgs(
            filters.Arg("name", "^"+pipeline+"_"+*modelID),
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

    optimizationFlags := worker.OptimizationFlags{
        "STREAM_PROTOCOL": "zeromq",
    }

    if err := w.Warm(ctx, pipeline, *modelID, worker.RunnerEndpoint{}, optimizationFlags); err != nil {
        slog.Error("Error warming container", slog.String("error", err.Error()))
        return
    }

    slog.Info("Warm container is up")

    req := worker.GenLiveVideoToVideoJSONRequestBody{
        ModelId:      modelID,
        SubscribeUrl: "tcp://172.17.0.1:5555",
        PublishUrl:   "tcp://172.17.0.1:5556",
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
            err := sendImages(ctx, *imagePath, *inputFps)
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
            err := receiveImages(ctx, *expOutputFps)
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
        slog.Info("Context done, waiting 10 sec for goroutines")
        time.Sleep(10 * time.Second)
        slog.Info("10 sec waiting done, stopping")
    case <-done:
        slog.Info("All goroutines finished")
    }

    w.Stop(ctx)
}
