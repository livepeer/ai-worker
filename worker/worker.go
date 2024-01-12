package worker

import (
	"context"
	"fmt"

	"github.com/docker/cli/opts"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/mount"
	"github.com/docker/docker/client"
	"github.com/docker/go-connections/nat"
)

const containerModelDir = "/models"
const containerPort = "8000/tcp"

var containerHostPorts = map[string]string{
	"text-to-image":  "8000",
	"image-to-image": "8001",
	"image-to-video": "8002",
}

type Worker struct {
	containerImageID string
	gpus             string
	modelDir         string
	outputDir        string

	dockerClient *client.Client
	containerIDs map[string]string
}

func NewWorker(containerImageID string, gpus string, modelDir string, outputDir string) (*Worker, error) {
	dockerClient, err := client.NewClientWithOpts(client.FromEnv, client.WithAPIVersionNegotiation())
	if err != nil {
		return nil, err
	}

	return &Worker{
		containerImageID: containerImageID,
		gpus:             gpus,
		modelDir:         modelDir,
		outputDir:        outputDir,
		dockerClient:     dockerClient,
		containerIDs:     make(map[string]string),
	}, nil
}

func (w *Worker) TextToImage(ctx context.Context, modelID string, req TextToImageJSONRequestBody) ([]string, error) {
	return nil, nil
}

func (w *Worker) ImageToImage(ctx context.Context, modelID string, req ImageToImageMultipartRequestBody) ([]string, error) {
	return nil, nil
}

func (w *Worker) ImageToVideo(ctx context.Context, modelID string, req ImageToVideoMultipartRequestBody) ([]string, error) {
	return nil, nil
}

func (w *Worker) Warm(ctx context.Context, containerName, modelID string) error {
	return w.warmContainer(ctx, containerName, modelID)
}

func (w *Worker) Stop(ctx context.Context, containerName string) error {
	containerID, ok := w.containerIDs[containerName]
	if !ok {
		return fmt.Errorf("container %v is not running", containerName)
	}

	// TODO: Handle if container fails to stop
	delete(w.containerIDs, containerName)

	return w.dockerClient.ContainerStop(ctx, containerID, container.StopOptions{})
}

func (w *Worker) warmContainer(ctx context.Context, containerName string, modelID string) error {
	if _, ok := w.containerIDs[containerName]; ok {
		return nil
	}

	// TODO: Pull image to ensure it exists

	containerConfig := &container.Config{
		Image: w.containerImageID,
		Env: []string{
			"PIPELINE=" + containerName,
			"MODEL_ID=" + modelID,
		},
		Volumes: map[string]struct{}{
			containerModelDir: {},
		},
		ExposedPorts: nat.PortSet{
			containerPort + "/tcp": struct{}{},
		},
	}

	gpuOpts := opts.GpuOpts{}
	gpuOpts.Set(w.gpus)

	hostConfig := &container.HostConfig{
		Resources: container.Resources{
			DeviceRequests: gpuOpts.Value(),
		},
		Mounts: []mount.Mount{
			{
				Type:   mount.TypeBind,
				Source: w.modelDir,
				Target: containerModelDir,
			},
		},
		PortBindings: nat.PortMap{
			containerPort: []nat.PortBinding{
				{
					HostIP:   "0.0.0.0",
					HostPort: containerHostPorts[containerName],
				},
			},
		},
	}

	resp, err := w.dockerClient.ContainerCreate(ctx, containerConfig, hostConfig, nil, nil, "")
	if err != nil {
		return err
	}

	if err := w.dockerClient.ContainerStart(ctx, resp.ID, types.ContainerStartOptions{}); err != nil {
		return err
	}

	statusCh, errCh := w.dockerClient.ContainerWait(ctx, resp.ID, container.WaitConditionNotRunning)
	select {
	case err := <-errCh:
		if err != nil {
			return err
		}
	case <-statusCh:
	}

	w.containerIDs[containerName] = resp.ID

	return nil
}
