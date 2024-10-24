package worker

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/docker/cli/opts"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/filters"
	"github.com/docker/docker/api/types/mount"
	"github.com/docker/docker/client"
	"github.com/docker/go-connections/nat"
)

const containerModelDir = "/models"
const containerPort = "8000/tcp"
const pollingInterval = 500 * time.Millisecond
const containerTimeout = 2 * time.Minute
const externalContainerTimeout = 2 * time.Minute
const optFlagsContainerTimeout = 5 * time.Minute
const containerRemoveTimeout = 30 * time.Second
const containerCreatorLabel = "creator"
const containerCreator = "ai-worker"

// This only works right now on a single GPU because if there is another container
// using the GPU we stop it so we don't have to worry about having enough ports
var containerHostPorts = map[string]string{
	"text-to-image":      "8000",
	"image-to-image":     "8100",
	"image-to-video":     "8200",
	"upscale":            "8300",
	"audio-to-text":      "8400",
	"llm":                "8500",
	"segment-anything-2": "8600",
	"image-to-text":      "8700",
	"sketch-to-image":    "8800",
}

// Mapping for per pipeline container images.
var pipelineToImage = map[string]string{
	"segment-anything-2": "livepeer/ai-runner:segment-anything-2",
}

type DockerManager struct {
	defaultImage string
	gpus         []string
	modelDir     string

	dockerClient *client.Client
	// gpu ID => container name
	gpuContainers map[string]string
	// container name => container
	containers map[string]*RunnerContainer
	mu         *sync.Mutex
}

func NewDockerManager(defaultImage string, gpus []string, modelDir string) (*DockerManager, error) {
	dockerClient, err := client.NewClientWithOpts(client.FromEnv, client.WithAPIVersionNegotiation())
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(context.Background(), containerTimeout)
	if err := removeExistingContainers(ctx, dockerClient); err != nil {
		cancel()
		return nil, err
	}
	cancel()

	return &DockerManager{
		defaultImage:  defaultImage,
		gpus:          gpus,
		modelDir:      modelDir,
		dockerClient:  dockerClient,
		gpuContainers: make(map[string]string),
		containers:    make(map[string]*RunnerContainer),
		mu:            &sync.Mutex{},
	}, nil
}

func (m *DockerManager) Warm(ctx context.Context, pipeline string, modelID string, optimizationFlags OptimizationFlags) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	_, err := m.createContainer(ctx, pipeline, modelID, true, optimizationFlags)
	return err
}

func (m *DockerManager) Stop(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	var stopContainerWg sync.WaitGroup
	for name, rc := range m.containers {
		stopContainerWg.Add(1)
		go func(containerID string) {
			defer stopContainerWg.Done()
			if err := dockerRemoveContainer(m.dockerClient, containerID); err != nil {
				slog.Error("Error removing managed container", slog.String("name", name), slog.String("id", containerID))
			}
		}(rc.ID)

		delete(m.gpuContainers, rc.GPU)
		delete(m.containers, name)
	}

	stopContainerWg.Wait()

	return nil
}

func (m *DockerManager) Borrow(ctx context.Context, pipeline, modelID string) (*RunnerContainer, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	for _, runner := range m.containers {
		if runner.Pipeline == pipeline && runner.ModelID == modelID {
			delete(m.containers, runner.Name)
			return runner, nil
		}
	}

	// The container does not exist so try to create it
	var err error
	// TODO: Optimization flags for dynamically loaded (borrowed) containers are not currently supported due to startup delays.
	rc, err := m.createContainer(ctx, pipeline, modelID, false, map[string]EnvValue{})
	if err != nil {
		return nil, err
	}

	// Remove container so it is unavailable until Return() is called
	delete(m.containers, rc.Name)
	return rc, nil
}

func (m *DockerManager) Return(rc *RunnerContainer) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.containers[rc.Name] = rc
}

// HasCapacity checks if an unused managed container exists or if a GPU is available for a new container.
func (m *DockerManager) HasCapacity(ctx context.Context, pipeline, modelID string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Check if unused managed container exists for the requested model.
	for _, rc := range m.containers {
		if rc.Pipeline == pipeline && rc.ModelID == modelID {
			return true
		}
	}

	// Check for available GPU to allocate for a new container for the requested model.
	_, err := m.allocGPU(ctx)
	return err == nil
}

func (m *DockerManager) createContainer(ctx context.Context, pipeline string, modelID string, keepWarm bool, optimizationFlags OptimizationFlags) (*RunnerContainer, error) {
	gpu, err := m.allocGPU(ctx)
	if err != nil {
		return nil, err
	}

	// NOTE: We currently allow only one container per GPU for each pipeline.
	containerHostPort := containerHostPorts[pipeline][:3] + gpu
	containerName := dockerContainerName(pipeline, modelID, containerHostPort)
	containerImage := m.defaultImage
	if pipelineSpecificImage, ok := pipelineToImage[pipeline]; ok {
		containerImage = pipelineSpecificImage
	}

	slog.Info("Starting managed container", slog.String("gpu", gpu), slog.String("name", containerName), slog.String("modelID", modelID), slog.String("containerImage", containerImage))

	// Add optimization flags as environment variables.
	envVars := []string{
		"PIPELINE=" + pipeline,
		"MODEL_ID=" + modelID,
	}
	for key, value := range optimizationFlags {
		envVars = append(envVars, key+"="+value.String())
	}

	containerConfig := &container.Config{
		Image: containerImage,
		Env:   envVars,
		Volumes: map[string]struct{}{
			containerModelDir: {},
		},
		ExposedPorts: nat.PortSet{
			containerPort: struct{}{},
		},
		Labels: map[string]string{
			containerCreatorLabel: containerCreator,
		},
	}

	gpuOpts := opts.GpuOpts{}
	gpuOpts.Set("device=" + gpu)

	hostConfig := &container.HostConfig{
		Resources: container.Resources{
			DeviceRequests: gpuOpts.Value(),
		},
		Mounts: []mount.Mount{
			{
				Type:   mount.TypeBind,
				Source: m.modelDir,
				Target: containerModelDir,
			},
		},
		PortBindings: nat.PortMap{
			containerPort: []nat.PortBinding{
				{
					HostIP:   "0.0.0.0",
					HostPort: containerHostPort,
				},
			},
		},
	}

	resp, err := m.dockerClient.ContainerCreate(ctx, containerConfig, hostConfig, nil, nil, containerName)
	if err != nil {
		return nil, err
	}

	cctx, cancel := context.WithTimeout(ctx, containerTimeout)
	if err := m.dockerClient.ContainerStart(cctx, resp.ID, container.StartOptions{}); err != nil {
		cancel()
		dockerRemoveContainer(m.dockerClient, resp.ID)
		return nil, err
	}
	cancel()

	cctx, cancel = context.WithTimeout(ctx, containerTimeout)
	if err := dockerWaitUntilRunning(cctx, m.dockerClient, resp.ID, pollingInterval); err != nil {
		cancel()
		dockerRemoveContainer(m.dockerClient, resp.ID)
		return nil, err
	}
	cancel()

	// Extend runner container timeout when optimization flags are used, as these
	// pipelines may require more startup time.
	runnerContainerTimeout := containerTimeout
	if len(optimizationFlags) > 0 {
		runnerContainerTimeout = optFlagsContainerTimeout
	}

	cfg := RunnerContainerConfig{
		Type:     Managed,
		Pipeline: pipeline,
		ModelID:  modelID,
		Endpoint: RunnerEndpoint{
			URL: "http://localhost:" + containerHostPort,
		},
		ID:               resp.ID,
		GPU:              gpu,
		KeepWarm:         keepWarm,
		containerTimeout: runnerContainerTimeout,
	}

	rc, err := NewRunnerContainer(ctx, cfg, containerName)
	if err != nil {
		dockerRemoveContainer(m.dockerClient, resp.ID)
		return nil, err
	}

	m.containers[containerName] = rc
	m.gpuContainers[gpu] = containerName

	return rc, nil
}

func (m *DockerManager) allocGPU(ctx context.Context) (string, error) {
	// Is there a GPU available?
	for _, gpu := range m.gpus {
		_, ok := m.gpuContainers[gpu]
		if !ok {
			return gpu, nil
		}
	}

	// Is there a GPU with an idle container?
	for _, gpu := range m.gpus {
		containerName := m.gpuContainers[gpu]
		// If the container exists in this map then it is idle and if it not marked as keep warm we remove it
		rc, ok := m.containers[containerName]
		if ok && !rc.KeepWarm {
			slog.Info("Removing managed container", slog.String("gpu", gpu), slog.String("name", containerName), slog.String("modelID", rc.ModelID))

			delete(m.gpuContainers, gpu)
			delete(m.containers, containerName)

			if err := dockerRemoveContainer(m.dockerClient, rc.ID); err != nil {
				return "", err
			}

			return gpu, nil
		}
	}

	return "", errors.New("insufficient capacity")
}

func removeExistingContainers(ctx context.Context, client *client.Client) error {
	filters := filters.NewArgs(filters.Arg("label", containerCreatorLabel+"="+containerCreator))
	containers, err := client.ContainerList(ctx, container.ListOptions{All: true, Filters: filters})
	if err != nil {
		return err
	}

	for _, c := range containers {
		slog.Info("Removing existing managed container", slog.String("name", c.Names[0]))
		if err := dockerRemoveContainer(client, c.ID); err != nil {
			return err
		}
	}

	return nil
}

// dockerContainerName generates a unique container name based on the pipeline, model ID, and an optional suffix.
func dockerContainerName(pipeline string, modelID string, suffix ...string) string {
	sanitizedModelID := strings.NewReplacer("/", "-", "_", "-").Replace(modelID)
	if len(suffix) > 0 {
		return fmt.Sprintf("%s_%s_%s", pipeline, sanitizedModelID, suffix[0])
	}
	return fmt.Sprintf("%s_%s", pipeline, sanitizedModelID)
}

func dockerRemoveContainer(client *client.Client, containerID string) error {
	ctx, cancel := context.WithTimeout(context.Background(), containerRemoveTimeout)
	if err := client.ContainerStop(ctx, containerID, container.StopOptions{}); err != nil {
		cancel()
		return err
	}
	cancel()

	ctx, cancel = context.WithTimeout(context.Background(), containerRemoveTimeout)
	defer cancel()
	return client.ContainerRemove(ctx, containerID, container.RemoveOptions{})
}

func dockerWaitUntilRunning(ctx context.Context, client *client.Client, containerID string, pollingInterval time.Duration) error {
	ticker := time.NewTicker(pollingInterval)
	defer ticker.Stop()

tickerLoop:
	for range ticker.C {
		select {
		case <-ctx.Done():
			return errors.New("timed out waiting for managed container")
		default:
			json, err := client.ContainerInspect(ctx, containerID)
			if err != nil {
				return err
			}

			if json.State.Running {
				break tickerLoop
			}
		}
	}

	return nil
}
