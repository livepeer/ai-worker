package worker

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strconv"
	"sync"
)

// EnvValue unmarshals JSON booleans as strings for compatibility with env variables.
type EnvValue string

// UnmarshalJSON converts JSON booleans to strings for EnvValue.
func (sb *EnvValue) UnmarshalJSON(b []byte) error {
	var boolVal bool
	err := json.Unmarshal(b, &boolVal)
	if err == nil {
		*sb = EnvValue(strconv.FormatBool(boolVal))
		return nil
	}

	var strVal string
	err = json.Unmarshal(b, &strVal)
	if err == nil {
		*sb = EnvValue(strVal)
	}

	return err
}

// String returns the string representation of the EnvValue.
func (sb EnvValue) String() string {
	return string(sb)
}

// handleJSONError is a global function that handles JSON errors and logs them.
func handleJSONError(pipeline string, statusCode int, jsonErr interface{}) error {
	if jsonErr != nil {
		val, err := json.Marshal(jsonErr)
		if err != nil {
			return err
		}
		slog.Error(fmt.Sprintf("%s container returned %d", pipeline, statusCode), slog.String("err", string(val)))
		return fmt.Errorf("%s container returned %d", pipeline, statusCode)
	}
	return nil
}

// OptimizationFlags is a map of optimization flags to be passed to the pipeline.
type OptimizationFlags map[string]EnvValue

type Worker struct {
	manager            *DockerManager
	externalContainers map[string]*RunnerContainer
	mu                 *sync.Mutex
}

func NewWorker(defaultImage string, gpus []string, modelDir string) (*Worker, error) {
	manager, err := NewDockerManager(defaultImage, gpus, modelDir)
	if err != nil {
		return nil, err
	}

	return &Worker{
		manager:            manager,
		externalContainers: make(map[string]*RunnerContainer),
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

	if resp.JSON200 != nil {
		return resp.JSON200, nil
	}

	if err := handleJSONError("text-to-image", 400, resp.JSON400); err != nil {
		return nil, err
	}
	if err := handleJSONError("text-to-image", 401, resp.JSON401); err != nil {
		return nil, err
	}
	if err := handleJSONError("text-to-image", 422, resp.JSON422); err != nil {
		return nil, err
	}
	if err := handleJSONError("text-to-image", 500, resp.JSON500); err != nil {
		return nil, err
	}

	slog.Error("text-to-image container returned unknown error. Please notify Livepeer AI team.")
	return nil, errors.New("text-to-image container returned unknown error. Please notify Livepeer AI team")
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

	if resp.JSON200 != nil {
		return resp.JSON200, nil
	}

	if err := handleJSONError("image-to-image", 400, resp.JSON400); err != nil {
		return nil, err
	}

	if err := handleJSONError("image-to-image", 401, resp.JSON401); err != nil {
		return nil, err
	}

	if err := handleJSONError("image-to-image", 422, resp.JSON422); err != nil {
		return nil, err
	}

	if err := handleJSONError("image-to-image", 500, resp.JSON500); err != nil {
		return nil, err
	}

	slog.Error("image-to-image container returned unknown error. Please notify Livepeer AI team.")
	return nil, errors.New("image-to-image container returned unknown error. Please notify Livepeer AI team")

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

	if resp.JSON200 != nil {
		return resp.JSON200, nil
	}

	if err := handleJSONError("image-to-video", 400, resp.JSON400); err != nil {
		return nil, err
	}
	if err := handleJSONError("image-to-video", 401, resp.JSON401); err != nil {
		return nil, err
	}
	if err := handleJSONError("image-to-video", 422, resp.JSON422); err != nil {
		return nil, err
	}
	if err := handleJSONError("image-to-video", 500, resp.JSON500); err != nil {
		return nil, err
	}

	slog.Error("image-to-video container returned unknown error. Please notify Livepeer AI team.")
	return nil, errors.New("image-to-video container returned unknown error. Please notify Livepeer AI team")
}

func (w *Worker) Upscale(ctx context.Context, req UpscaleMultipartRequestBody) (*ImageResponse, error) {
	c, err := w.borrowContainer(ctx, "upscale", *req.ModelId)
	if err != nil {
		return nil, err
	}
	defer w.returnContainer(c)

	var buf bytes.Buffer
	mw, err := NewUpscaleMultipartWriter(&buf, req)
	if err != nil {
		return nil, err
	}

	resp, err := c.Client.UpscaleWithBodyWithResponse(ctx, mw.FormDataContentType(), &buf)
	if err != nil {
		return nil, err
	}

	if resp.JSON200 != nil {
		return resp.JSON200, nil
	}

	if err := handleJSONError("upscale", 400, resp.JSON400); err != nil {
		return nil, err
	}
	if err := handleJSONError("upscale", 401, resp.JSON401); err != nil {
		return nil, err
	}
	if err := handleJSONError("upscale", 422, resp.JSON422); err != nil {
		return nil, err
	}
	if err := handleJSONError("upscale", 500, resp.JSON500); err != nil {
		return nil, err
	}

	slog.Error("upscale container returned unknown error. Please notify Livepeer AI team.")
	return nil, errors.New("upscale container returned unknown error. Please notify Livepeer AI team")
}

func (w *Worker) AudioToText(ctx context.Context, req AudioToTextMultipartRequestBody) (*TextResponse, error) {
	c, err := w.borrowContainer(ctx, "audio-to-text", *req.ModelId)
	if err != nil {
		return nil, err
	}
	defer w.returnContainer(c)

	var buf bytes.Buffer
	mw, err := NewAudioToTextMultipartWriter(&buf, req)
	if err != nil {
		return nil, err
	}

	resp, err := c.Client.AudioToTextWithBodyWithResponse(ctx, mw.FormDataContentType(), &buf)
	if err != nil {
		return nil, err
	}

	if resp.JSON200 != nil {
		return resp.JSON200, nil
	}

	if err := handleJSONError("audio-to-text", 400, resp.JSON400); err != nil {
		return nil, err
	}
	if err := handleJSONError("audio-to-text", 401, resp.JSON401); err != nil {
		return nil, err
	}
	if err := handleJSONError("audio-to-text", 413, resp.JSON413); err != nil {
		return nil, fmt.Errorf("%s: file too large; max file size is 50MB", err.Error())
	}
	if err := handleJSONError("audio-to-text", 415, resp.JSON415); err != nil {
		return nil, err
	}
	if err := handleJSONError("audio-to-text", 422, resp.JSON422); err != nil {
		return nil, err
	}
	if err := handleJSONError("audio-to-text", 500, resp.JSON500); err != nil {
		return nil, err
	}

	slog.Error("audio-to-text container returned unknown error. Please notify Livepeer AI team.")
	return nil, errors.New("audio-to-text container returned unknown error. Please notify Livepeer AI team")
}

func (w *Worker) SegmentAnything2(ctx context.Context, req SegmentAnything2MultipartRequestBody) (*MasksResponse, error) {
	c, err := w.borrowContainer(ctx, "segment-anything-2", *req.ModelId)
	if err != nil {
		return nil, err
	}
	defer w.returnContainer(c)

	var buf bytes.Buffer
	mw, err := NewSegmentAnything2MultipartWriter(&buf, req)
	if err != nil {
		return nil, err
	}

	resp, err := c.Client.SegmentAnything2WithBodyWithResponse(ctx, mw.FormDataContentType(), &buf)
	if err != nil {
		return nil, err
	}

	if resp.JSON200 != nil {
		return resp.JSON200, nil
	}

	if err := handleJSONError("segment-anything-2", 400, resp.JSON400); err != nil {
		return nil, err
	}
	if err := handleJSONError("segment-anything-2", 401, resp.JSON401); err != nil {
		return nil, err
	}
	if err := handleJSONError("segment-anything-2", 422, resp.JSON422); err != nil {
		return nil, err
	}
	if err := handleJSONError("segment-anything-2", 500, resp.JSON500); err != nil {
		return nil, err
	}

	slog.Error("segment-anything-2 container returned unknown error. Please notify Livepeer AI team.")
	return nil, errors.New("segment-anything-2 container returned unknown error. Please notify Livepeer AI team")
}

func (w *Worker) Warm(ctx context.Context, pipeline string, modelID string, endpoint RunnerEndpoint, optimizationFlags OptimizationFlags) error {
	if endpoint.URL == "" {
		return w.manager.Warm(ctx, pipeline, modelID, optimizationFlags)
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	cfg := RunnerContainerConfig{
		Type:             External,
		Pipeline:         pipeline,
		ModelID:          modelID,
		Endpoint:         endpoint,
		containerTimeout: externalContainerTimeout,
	}
	rc, err := NewRunnerContainer(ctx, cfg, endpoint.URL)
	if err != nil {
		return err
	}

	name := dockerContainerName(pipeline, modelID, endpoint.URL)
	slog.Info("Starting external container", slog.String("name", name), slog.String("modelID", modelID))
	w.externalContainers[name] = rc

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

// HasCapacity returns true if the worker has capacity for the given pipeline and model ID.
func (w *Worker) HasCapacity(pipeline, modelID string) bool {
	w.mu.Lock()
	defer w.mu.Unlock()

	// Check if we have capacity for external containers.
	for _, rc := range w.externalContainers {
		if rc.Pipeline == pipeline && rc.ModelID == modelID {
			return true
		}
	}

	// Check if we have capacity for managed containers.
	return w.manager.HasCapacity(context.Background(), pipeline, modelID)
}

func (w *Worker) borrowContainer(ctx context.Context, pipeline, modelID string) (*RunnerContainer, error) {
	w.mu.Lock()

	for _, rc := range w.externalContainers {
		if rc.Pipeline == pipeline && rc.ModelID == modelID {
			w.mu.Unlock()
			// Assume external containers can handle concurrent in-flight requests.
			return rc, nil
		}
	}

	w.mu.Unlock()

	return w.manager.Borrow(ctx, pipeline, modelID)
}

func (w *Worker) returnContainer(rc *RunnerContainer) {
	switch rc.Type {
	case Managed:
		w.manager.Return(rc)
	case External:
		// Noop because we allow concurrent in-flight requests for external containers
	}
}
