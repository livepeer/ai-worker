package worker

import (
	"context"
	"errors"
	"time"

	"github.com/deepmap/oapi-codegen/v2/pkg/securityprovider"
)

type RunnerContainerType int

const (
	Managed RunnerContainerType = iota
	External
)

type RunnerContainer struct {
	RunnerContainerConfig
	Name     string
	Capacity int
	Client   *ClientWithResponses
}

type RunnerEndpoint struct {
	URL   string
	Token string
}

type RunnerContainerConfig struct {
	Type     RunnerContainerType
	Pipeline string
	ModelID  string
	Endpoint RunnerEndpoint

	// For managed containers only
	ID               string
	GPU              string
	KeepWarm         bool
	containerTimeout time.Duration
}

func NewRunnerContainer(ctx context.Context, cfg RunnerContainerConfig, name string) (*RunnerContainer, error) {
	// Ensure that timeout is set to a non-zero value.
	timeout := cfg.containerTimeout
	if timeout == 0 {
		timeout = containerTimeout
	}

	var opts []ClientOption
	if cfg.Endpoint.Token != "" {
		bearerTokenProvider, err := securityprovider.NewSecurityProviderBearerToken(cfg.Endpoint.Token)
		if err != nil {
			return nil, err
		}

		opts = append(opts, WithRequestEditorFn(bearerTokenProvider.Intercept))
	}

	client, err := NewClientWithResponses(cfg.Endpoint.URL, opts...)
	if err != nil {
		return nil, err
	}

	cctx, cancel := context.WithTimeout(context.Background(), cfg.containerTimeout)
	if err := runnerWaitUntilReady(cctx, client, pollingInterval); err != nil {
		cancel()
		return nil, err
	}
	cancel()

	return &RunnerContainer{
		RunnerContainerConfig: cfg,
		Name:                  name,
		Capacity:              1,
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
