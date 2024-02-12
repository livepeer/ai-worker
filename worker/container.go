package worker

import (
	"context"
	"errors"
	"time"
)

type RunnerContainerType int

const (
	Managed RunnerContainerType = iota
	External
)

type RunnerContainer struct {
	RunnerContainerConfig

	Client *ClientWithResponses
}

type RunnerContainerConfig struct {
	Type     RunnerContainerType
	Pipeline string
	ModelID  string
	Endpoint string

	// For managed containers only
	ID       string
	GPU      string
	KeepWarm bool
}

func NewRunnerContainer(ctx context.Context, cfg RunnerContainerConfig) (*RunnerContainer, error) {
	client, err := NewClientWithResponses(cfg.Endpoint)
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(ctx, containerTimeout)
	if err := runnerWaitUntilReady(ctx, client, pollingInterval); err != nil {
		cancel()
		return nil, err
	}
	cancel()

	return &RunnerContainer{
		RunnerContainerConfig: cfg,
		Client:                client,
	}, nil
}

func runnerWaitUntilReady(ctx context.Context, client *ClientWithResponses, pollingInterval time.Duration) error {
	ticker := time.NewTicker(pollingInterval)
	defer ticker.Stop()

tickerLoop:
	for range ticker.C {
		select {
		case <-ctx.Done():
			return errors.New("timed out waiting for runner")
		default:
			if _, err := client.HealthWithResponse(ctx); err == nil {
				break tickerLoop
			}
		}
	}

	return nil
}
