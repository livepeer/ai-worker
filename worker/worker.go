package worker

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"mime/multipart"
	"sync"
	"time"

	"github.com/docker/cli/opts"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/filters"
	"github.com/docker/docker/api/types/mount"
	"github.com/docker/docker/client"
	"github.com/docker/go-connections/nat"
)

const containerModelDir = "/models"
const containerPort = "8000/tcp"
const pollingInterval = 500 * time.Millisecond

var containerHostPorts = map[string]string{
	"text-to-image":  "8000",
	"image-to-image": "8001",
	"image-to-video": "8002",
}

type RunnerContainer struct {
	ID      string
	Client  *ClientWithResponses
	ModelID string
	GPU     string
}

type Worker struct {
	containerImageID string
	gpus             []string
	gpuLoad          map[string]int
	modelDir         string

	dockerClient *client.Client
	containers   map[string]*RunnerContainer
	mu           *sync.Mutex
}

func NewWorker(containerImageID string, gpus []string, modelDir string) (*Worker, error) {
	dockerClient, err := client.NewClientWithOpts(client.FromEnv, client.WithAPIVersionNegotiation())
	if err != nil {
		return nil, err
	}

	return &Worker{
		containerImageID: containerImageID,
		gpus:             gpus,
		gpuLoad:          make(map[string]int),
		modelDir:         modelDir,
		dockerClient:     dockerClient,
		containers:       make(map[string]*RunnerContainer),
		mu:               &sync.Mutex{},
	}, nil
}

func (w *Worker) TextToImage(ctx context.Context, req TextToImageJSONRequestBody) (*ImageResponse, error) {
	c, err := w.getWarmContainer(ctx, "text-to-image", *req.ModelId)
	if err != nil {
		return nil, err
	}

	resp, err := c.Client.TextToImageWithResponse(ctx, req)
	if err != nil {
		return nil, err
	}

	if resp.JSON422 != nil {
		// TODO: Handle JSON422 struct
		return nil, errors.New("text-to-image container returned 422")
	}

	return resp.JSON200, nil
}

func (w *Worker) ImageToImage(ctx context.Context, req ImageToImageMultipartRequestBody) (*ImageResponse, error) {
	c, err := w.getWarmContainer(ctx, "image-to-image", *req.ModelId)
	if err != nil {
		return nil, err
	}

	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	writer, err := mw.CreateFormFile("image", req.Image.Filename())
	if err != nil {
		return nil, err
	}
	imageSize := req.Image.FileSize()
	imageRdr, err := req.Image.Reader()
	if err != nil {
		return nil, err
	}
	copied, err := io.Copy(writer, imageRdr)
	if err != nil {
		return nil, err
	}
	if copied != imageSize {
		return nil, fmt.Errorf("failed to copy image to multipart request imageBytes=%v copiedBytes=%v", imageSize, copied)
	}

	if err := mw.WriteField("prompt", req.Prompt); err != nil {
		return nil, err
	}
	if err := mw.WriteField("model_id", *req.ModelId); err != nil {
		return nil, err
	}

	if err := mw.Close(); err != nil {
		return nil, err
	}

	resp, err := c.Client.ImageToImageWithBodyWithResponse(ctx, mw.FormDataContentType(), &buf)
	if err != nil {
		return nil, err
	}

	if resp.JSON422 != nil {
		// TODO: Handle JSON422 struct
		return nil, errors.New("image-to-image container returned 422")
	}

	return resp.JSON200, nil
}

func (w *Worker) ImageToVideo(ctx context.Context, req ImageToVideoMultipartRequestBody) (*VideoResponse, error) {
	c, err := w.getWarmContainer(ctx, "image-to-video", *req.ModelId)
	if err != nil {
		return nil, err
	}

	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	writer, err := mw.CreateFormFile("image", req.Image.Filename())
	if err != nil {
		return nil, err
	}
	imageSize := req.Image.FileSize()
	imageRdr, err := req.Image.Reader()
	if err != nil {
		return nil, err
	}
	copied, err := io.Copy(writer, imageRdr)
	if err != nil {
		return nil, err
	}
	if copied != imageSize {
		return nil, fmt.Errorf("failed to copy image to multipart request imageBytes=%v copiedBytes=%v", imageSize, copied)
	}

	if err := mw.WriteField("model_id", *req.ModelId); err != nil {
		return nil, err
	}

	if err := mw.Close(); err != nil {
		return nil, err
	}

	resp, err := c.Client.ImageToVideoWithBodyWithResponse(ctx, mw.FormDataContentType(), &buf)
	if err != nil {
		return nil, err
	}

	if resp.JSON422 != nil {
		// TODO: Handle JSON422 struct
		return nil, errors.New("image-to-video container returned 422")
	}

	return resp.JSON200, nil
}

func (w *Worker) Warm(ctx context.Context, containerName, modelID string) error {
	_, err := w.getWarmContainer(ctx, containerName, modelID)
	return err
}

func (w *Worker) Stop(ctx context.Context, containerName string) error {
	c, ok := w.containers[containerName]
	if !ok {
		return fmt.Errorf("container %v is not running", containerName)
	}

	// TODO: Handle if container fails to stop or be removed
	delete(w.containers, containerName)

	if err := w.dockerClient.ContainerStop(ctx, c.ID, container.StopOptions{}); err != nil {
		return err
	}

	// Is there a reason to not remove the container?
	return w.dockerClient.ContainerRemove(ctx, c.ID, types.ContainerRemoveOptions{})
}

func (w *Worker) getWarmContainer(ctx context.Context, containerName string, modelID string) (*RunnerContainer, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	filters := filters.NewArgs(filters.Arg("name", "^"+containerName+"$"), filters.Arg("status", "running"))
	containers, err := w.dockerClient.ContainerList(ctx, types.ContainerListOptions{Filters: filters})
	if err != nil {
		return nil, err
	}

	if len(containers) > 0 {
		c := containers[0]
		rc, ok := w.containers[containerName]
		if ok {
			// Return the running container if it is tracked by the worker already
			if rc.ID == c.ID {
				slog.Info("Using warm container", slog.String("gpu", rc.GPU), slog.String("name", containerName), slog.String("modelID", modelID))
				return rc, nil
			}

			// The worker is tracking a different container with the same name
			// We'll stop and remove this container so the worker can start and properly track a newly started container
			w.gpuLoad[rc.GPU] -= 1
			delete(w.containers, containerName)
		}

		slog.Info("Removing untracked container", slog.String("name", containerName))

		if err := dockerRemoveContainer(ctx, w.dockerClient, c.ID); err != nil {
			return nil, err
		}
	}

	// Get next available GPU
	gpu, load := w.leastLoadedGPU()
	if load > 0 {
		// For simplicity, if there are already containers using this GPU, stop and remove them
		// In the future, we can more intelligently decide when to stop containers that are using this GPU (i.e. based on low VRAM)
		for name, rc := range w.containers {
			if rc.GPU == gpu {
				slog.Info("Removing container", slog.String("gpu", gpu), slog.String("name", name), slog.String("modelID", rc.ModelID))

				w.gpuLoad[rc.GPU] -= 1
				delete(w.containers, name)

				if err := dockerRemoveContainer(ctx, w.dockerClient, rc.ID); err != nil {
					return nil, err
				}
			}
		}
	}

	slog.Info("Starting container", slog.String("gpu", gpu), slog.String("name", containerName), slog.String("modelID", modelID))

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
			containerPort: struct{}{},
		},
	}

	gpuOpts := opts.GpuOpts{}
	gpuOpts.Set(gpu)

	containerHostPort := containerHostPorts[containerName]
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
					HostPort: containerHostPort,
				},
			},
		},
	}

	resp, err := w.dockerClient.ContainerCreate(ctx, containerConfig, hostConfig, nil, nil, containerName)
	if err != nil {
		return nil, err
	}

	if err := w.dockerClient.ContainerStart(ctx, resp.ID, types.ContainerStartOptions{}); err != nil {
		return nil, err
	}

	// TODO: Add timeout with context
	if err := dockerWaitUntilRunning(ctx, w.dockerClient, resp.ID, pollingInterval); err != nil {
		return nil, err
	}

	client, err := NewClientWithResponses("http://localhost:" + containerHostPort)
	if err != nil {
		return nil, err
	}

	// TODO: Add timeout with context
	if err := runnerWaitUntilReady(ctx, client, pollingInterval); err != nil {
		return nil, err
	}

	rc := &RunnerContainer{
		ID:      resp.ID,
		Client:  client,
		ModelID: modelID,
		GPU:     gpu,
	}

	w.containers[containerName] = rc
	w.gpuLoad[gpu] += 1

	return rc, nil
}

func (w *Worker) leastLoadedGPU() (string, int) {
	minGPU := "0"
	minLoad := math.MaxInt64

	for _, gpu := range w.gpus {
		if w.gpuLoad[gpu] < minLoad {
			minLoad = w.gpuLoad[gpu]
			minGPU = gpu
		}
	}

	return minGPU, minLoad
}

func dockerRemoveContainer(ctx context.Context, client *client.Client, containerID string) error {
	if err := client.ContainerStop(ctx, containerID, container.StopOptions{}); err != nil {
		return err
	}

	return client.ContainerRemove(ctx, containerID, types.ContainerRemoveOptions{})
}

func dockerWaitUntilRunning(ctx context.Context, client *client.Client, containerID string, pollingInterval time.Duration) error {
	ticker := time.NewTicker(pollingInterval)
	defer ticker.Stop()

tickerLoop:
	for range ticker.C {
		select {
		case <-ctx.Done():
			return errors.New("timed out waiting for container")
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
