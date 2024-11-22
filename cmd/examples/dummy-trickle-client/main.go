package main

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"sync"
	"time"
	"trickle"
)

type ImagePublisher struct {
	publisher *trickle.TricklePublisher
	imagePath string
	fps       int
}

type ImageSubscriber struct {
	subscriber *trickle.TrickleSubscriber
	startTime  time.Time
	frameCount int
	mu         sync.Mutex
}

func (ip *ImagePublisher) publishImages(ctx context.Context, wg *sync.WaitGroup) {
	defer wg.Done()

	// Read image file
	imageData, err := os.ReadFile(ip.imagePath)
	if err != nil {
		slog.Error("Failed to read image", "error", err)
		return
	}

	interval := time.Second / time.Duration(ip.fps)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// Convert []byte to io.Reader using bytes.NewReader
			reader := bytes.NewReader(imageData)
			err := ip.publisher.Write(reader)
			if err != nil {
				slog.Error("Failed to publish image", "error", err)
				return
			}
		}
	}
}

func (is *ImageSubscriber) subscribe(ctx context.Context, wg *sync.WaitGroup) {
	defer wg.Done()

	is.startTime = time.Now()
	for {
		select {
		case <-ctx.Done():
			return
		default:
			resp, err := is.subscriber.Read()
			if err != nil {
				slog.Error("Failed to read segment", "error", err)
				continue
			}

			is.mu.Lock()
			is.frameCount++
			elapsed := time.Since(is.startTime).Seconds()
			currentFPS := float64(is.frameCount) / elapsed
			slog.Info("FPS Stats",
				"frames", is.frameCount,
				"elapsed_seconds", fmt.Sprintf("%.2f", elapsed),
				"current_fps", fmt.Sprintf("%.2f", currentFPS))
			is.mu.Unlock()

			resp.Body.Close()
		}
	}
}

func main() {
	imagePath := flag.String("image", "", "Path to image file")
	fps := flag.Int("fps", 30, "Target frames per second")
	publishURL := flag.String("publishurl", "http://localhost:2865/image-send", "URL to publish images")
	subscribeURL := flag.String("subscribeurl", "http://localhost:2865/image-recv", "URL to subscribe to images")
	flag.Parse()

	if *imagePath == "" {
		slog.Error("Image path is required")
		flag.Usage()
		os.Exit(1)
	}

	// Create publisher
	pub, err := trickle.NewTricklePublisher(*publishURL)
	if err != nil {
		slog.Error("Failed to create publisher", "error", err)
		os.Exit(1)
	}
	defer pub.Close()

	// Create subscriber
	sub := trickle.NewTrickleSubscriber(*subscribeURL)

	imagePublisher := &ImagePublisher{
		publisher: pub,
		imagePath: *imagePath,
		fps:       *fps,
	}

	imageSubscriber := &ImageSubscriber{
		subscriber: sub,
	}

	// Create context for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create WaitGroup for goroutines
	var wg sync.WaitGroup
	wg.Add(2)

	// Start publisher goroutine
	go imagePublisher.publishImages(ctx, &wg)

	// Start subscriber goroutine
	go imageSubscriber.subscribe(ctx, &wg)

	// Wait for interrupt signal
	slog.Info("Running... Press Ctrl+C to stop")
	select {}
}
