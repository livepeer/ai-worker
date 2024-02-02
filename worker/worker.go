package worker

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"mime/multipart"
	"strings"
	"sync"
	"time"

	"github.com/docker/cli/opts"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/mount"
	"github.com/docker/docker/client"
	"github.com/docker/go-connections/nat"
)

const containerModelDir = "/models"
const containerPort = "8000/tcp"
const pollingInterval = 500 * time.Millisecond
const containerTimeout = 30 * time.Second

// This only works right now on a single GPU because if there is another container
// using the GPU we stop it so we don't have to worry about having enough ports
var containerHostPorts = map[string]string{
	"text-to-image":  "8000",
	"image-to-image": "8001",
	"image-to-video": "8002",
}

type RunnerContainer struct {
	ID       string
	Client   *ClientWithResponses
	Pipeline string
	ModelID  string
	GPU      string
}

type Worker struct {
	containerImageID string
	gpus             []string
	modelDir         string

	dockerClient *client.Client
	// gpu ID => container name
	gpuContainers map[string]string
	// container name => container
	containers map[string]*RunnerContainer
	mu         *sync.Mutex
}

func NewWorker(containerImageID string, gpus []string, modelDir string) (*Worker, error) {
	dockerClient, err := client.NewClientWithOpts(client.FromEnv, client.WithAPIVersionNegotiation())
	if err != nil {
		return nil, err
	}

	return &Worker{
		containerImageID: containerImageID,
		gpus:             gpus,
		modelDir:         modelDir,
		dockerClient:     dockerClient,
		gpuContainers:    make(map[string]string),
		containers:       make(map[string]*RunnerContainer),
		mu:               &sync.Mutex{},
	}, nil
}

func (w *Worker) TextToImage(ctx context.Context, req TextToImageJSONRequestBody) (*ImageResponse, error) {
	c, err := w.borrowContainer(ctx, "text-to-image", *req.ModelId)
	if err != nil {
		return nil, err
	}
	defer w.returnContainer(c)

	resp, err := c.Client.TextToImageWithResponse(ctx, req)
	if err != nil {
		return nil, err
	}

	if resp.JSON422 != nil {
		val, err := json.Marshal(resp.JSON422)
		if err != nil {
			return nil, err
		}
		slog.Error("text-to-image container returned 422", slog.String("err", string(val)))
		return nil, errors.New("text-to-image container returned 422")
	}

	if resp.JSON400 != nil {
		val, err := json.Marshal(resp.JSON400)
		if err != nil {
			return nil, err
		}
		slog.Error("text-to-image container returned 400", slog.String("err", string(val)))
		return nil, errors.New("text-to-image container returned 400")
	}

	if resp.JSON500 != nil {
		val, err := json.Marshal(resp.JSON500)
		if err != nil {
			return nil, err
		}
		slog.Error("text-to-image container returned 500", slog.String("err", string(val)))
		return nil, errors.New("text-to-image container returned 500")
	}

	return resp.JSON200, nil
}

func (w *Worker) ImageToImage(ctx context.Context, req ImageToImageMultipartRequestBody) (*ImageResponse, error) {
	c, err := w.borrowContainer(ctx, "image-to-image", *req.ModelId)
	if err != nil {
		return nil, err
	}
	defer w.returnContainer(c)

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
		val, err := json.Marshal(resp.JSON422)
		if err != nil {
			return nil, err
		}
		slog.Error("image-to-image container returned 422", slog.String("err", string(val)))
		return nil, errors.New("image-to-image container returned 422")
	}

	if resp.JSON400 != nil {
		val, err := json.Marshal(resp.JSON400)
		if err != nil {
			return nil, err
		}
		slog.Error("image-to-image container returned 400", slog.String("err", string(val)))
		return nil, errors.New("image-to-image container returned 400")
	}

	if resp.JSON500 != nil {
		val, err := json.Marshal(resp.JSON500)
		if err != nil {
			return nil, err
		}
		slog.Error("image-to-image container returned 500", slog.String("err", string(val)))
		return nil, errors.New("image-to-image container returned 500")
	}

	return resp.JSON200, nil
}

func (w *Worker) ImageToVideo(ctx context.Context, req ImageToVideoMultipartRequestBody) (*VideoResponse, error) {
	c, err := w.borrowContainer(ctx, "image-to-video", *req.ModelId)
	if err != nil {
		return nil, err
	}
	defer w.returnContainer(c)

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
		val, err := json.Marshal(resp.JSON422)
		if err != nil {
			return nil, err
		}
		slog.Error("image-to-video container returned 422", slog.String("err", string(val)))
		return nil, errors.New("image-to-video container returned 422")
	}

	if resp.JSON400 != nil {
		val, err := json.Marshal(resp.JSON400)
		if err != nil {
			return nil, err
		}
		slog.Error("image-to-video container returned 400", slog.String("err", string(val)))
		return nil, errors.New("image-to-video container returned 400")
	}

	if resp.JSON500 != nil {
		val, err := json.Marshal(resp.JSON500)
		if err != nil {
			return nil, err
		}
		slog.Error("image-to-video container returned 500", slog.String("err", string(val)))
		return nil, errors.New("image-to-video container returned 500")
	}

	return resp.JSON200, nil
}

func (w *Worker) Warm(ctx context.Context, containerName, modelID string) error {
	_, err := w.createContainer(ctx, containerName, modelID)
	return err
}

func (w *Worker) Stop(ctx context.Context) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	var stopContainerWg sync.WaitGroup
	for name, rc := range w.containers {
		stopContainerWg.Add(1)
		go func(containerID string) {
			defer stopContainerWg.Done()
			if err := dockerRemoveContainer(ctx, w.dockerClient, containerID); err != nil {
				slog.Error("Error removing container", slog.String("name", name), slog.String("id", containerID))
			}
		}(rc.ID)

		delete(w.gpuContainers, rc.GPU)
		delete(w.containers, name)
	}

	stopContainerWg.Wait()

	return nil
}

func (w *Worker) borrowContainer(ctx context.Context, pipeline, modelID string) (*RunnerContainer, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	containerName := dockerContainerName(pipeline, modelID)
	rc, ok := w.containers[containerName]
	if !ok {
		// The container does not exist so try to create it
		var err error
		rc, err = w.createContainer(ctx, pipeline, modelID)
		if err != nil {
			return nil, err
		}
	}

	// Remove container so it is unavailable until returnContainer is called
	delete(w.containers, containerName)
	return rc, nil
}

func (w *Worker) returnContainer(rc *RunnerContainer) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.containers[dockerContainerName(rc.Pipeline, rc.ModelID)] = rc
}

func (w *Worker) createContainer(ctx context.Context, pipeline string, modelID string) (*RunnerContainer, error) {
	containerName := dockerContainerName(pipeline, modelID)

	gpu, err := w.allocGPU(ctx)
	if err != nil {
		return nil, err
	}

	slog.Info("Starting container", slog.String("gpu", gpu), slog.String("name", containerName), slog.String("modelID", modelID))

	containerConfig := &container.Config{
		Image: w.containerImageID,
		Env: []string{
			"PIPELINE=" + pipeline,
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
	gpuOpts.Set("device=" + gpu)

	containerHostPort := containerHostPorts[pipeline]
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

	cctx, cancel := context.WithTimeout(ctx, containerTimeout)
	if err := dockerWaitUntilRunning(cctx, w.dockerClient, resp.ID, pollingInterval); err != nil {
		cancel()
		return nil, err
	}
	cancel()

	client, err := NewClientWithResponses("http://localhost:" + containerHostPort)
	if err != nil {
		return nil, err
	}

	cctx, cancel = context.WithTimeout(ctx, containerTimeout)
	if err := runnerWaitUntilReady(cctx, client, pollingInterval); err != nil {
		cancel()
		return nil, err
	}
	cancel()

	rc := &RunnerContainer{
		ID:       resp.ID,
		Client:   client,
		Pipeline: pipeline,
		ModelID:  modelID,
		GPU:      gpu,
	}

	w.containers[containerName] = rc
	w.gpuContainers[gpu] = containerName

	return rc, nil
}

func (w *Worker) allocGPU(ctx context.Context) (string, error) {
	// Is there a GPU available?
	for _, gpu := range w.gpus {
		_, ok := w.gpuContainers[gpu]
		if !ok {
			return gpu, nil
		}
	}

	// Is there a GPU with an idle container?
	for _, gpu := range w.gpus {
		containerName := w.gpuContainers[gpu]
		// If the container exists in this map then it is idle and we remove it
		rc, ok := w.containers[containerName]
		if ok {
			slog.Info("Removing container", slog.String("gpu", gpu), slog.String("name", containerName), slog.String("modelID", rc.ModelID))

			delete(w.gpuContainers, gpu)
			delete(w.containers, containerName)

			if err := dockerRemoveContainer(ctx, w.dockerClient, rc.ID); err != nil {
				return "", err
			}

			return gpu, nil
		}
	}

	return "", errors.New("insufficient capacity")
}

func dockerContainerName(pipeline string, modelID string) string {
	// text-to-image, stabilityai/sd-turbo -> text-to-image_stabilityai_sd-turbo
	// image-to-video, stabilityai/stable-video-diffusion-img2vid-xt -> image-to-video_stabilityai_stable-video-diffusion-img2vid-xt
	return strings.ReplaceAll(pipeline+"_"+modelID, "/", "_")
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
