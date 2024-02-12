package worker

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"sync"
)

type Worker struct {
	manager *DockerManager

	externalContainers map[string][]*RunnerContainer
	mu                 *sync.Mutex
}

func NewWorker(containerImageID string, gpus []string, modelDir string) (*Worker, error) {
	manager, err := NewDockerManager(containerImageID, gpus, modelDir)
	if err != nil {
		return nil, err
	}

	return &Worker{
		manager:            manager,
		externalContainers: make(map[string][]*RunnerContainer),
		mu:                 &sync.Mutex{},
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
	mw, err := NewImageToImageMultipartWriter(&buf, req)
	if err != nil {
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
	mw, err := NewImageToVideoMultipartWriter(&buf, req)
	if err != nil {
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

func (w *Worker) Warm(ctx context.Context, pipeline, modelID, endpoint string) error {
	if endpoint == "" {
		return w.manager.Warm(ctx, pipeline, modelID)
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	cfg := RunnerContainerConfig{
		Type:     External,
		Pipeline: pipeline,
		ModelID:  modelID,
		Endpoint: endpoint,
	}
	rc, err := NewRunnerContainer(ctx, cfg)
	if err != nil {
		return err
	}

	name := dockerContainerName(pipeline, modelID)
	slog.Info("Starting external container", slog.String("name", name), slog.String("modelID", modelID))
	w.externalContainers[name] = append(w.externalContainers[name], rc)

	return nil
}

func (w *Worker) Stop(ctx context.Context) error {
	if err := w.manager.Stop(ctx); err != nil {
		return err
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	for name := range w.externalContainers {
		delete(w.externalContainers, name)
	}

	return nil
}

func (w *Worker) borrowContainer(ctx context.Context, pipeline, modelID string) (*RunnerContainer, error) {
	w.mu.Lock()

	name := dockerContainerName(pipeline, modelID)
	containers := w.externalContainers[name]
	if len(containers) > 0 {
		rc := containers[0]
		w.externalContainers[name] = containers[1:]
		w.mu.Unlock()
		return rc, nil
	}

	w.mu.Unlock()

	return w.manager.Borrow(ctx, pipeline, modelID)
}

func (w *Worker) returnContainer(rc *RunnerContainer) {
	switch rc.Type {
	case Managed:
		w.manager.Return(rc)
	case External:
		w.mu.Lock()
		name := dockerContainerName(rc.Pipeline, rc.ModelID)
		w.externalContainers[name] = append(w.externalContainers[name], rc)
		w.mu.Unlock()
	}
}
