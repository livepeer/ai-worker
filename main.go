package main

import (
	"context"
	"ffai-worker/executor"
	"log/slog"
	"time"
)

func main() {
	executor := executor.NewExecutor(executor.Config{
		CogImage:    "sdxl-turbo",
		GPUs:        "all",
		IdleTimeout: 1 * time.Minute,
	})

	if err := executor.Start(context.Background()); err != nil {
		slog.Error("Error starting executor", slog.String("error", err.Error()))
		return
	}

	inputs := map[string]string{
		"prompt": "Miyamoto Musashi on a horse",
	}
	if err := executor.Execute(inputs, "output.png"); err != nil {
		slog.Error("Error executing job", slog.String("error", err.Error()))
		return
	}
	if err := executor.Execute(inputs, "output1.png"); err != nil {
		slog.Error("Error executing job", slog.String("error", err.Error()))
		return
	}
	if err := executor.Execute(inputs, "output2.png"); err != nil {
		slog.Error("Error executing job", slog.String("error", err.Error()))
		return
	}
}
