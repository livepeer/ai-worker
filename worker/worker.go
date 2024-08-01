package worker

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
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

// OptimizationFlags is a map of optimization flags to be passed to the pipeline.
type OptimizationFlags map[string]EnvValue

type Worker struct {
	manager *DockerManager

	externalContainers map[string]*RunnerContainer
	mu                 *sync.Mutex
}

func NewWorker(containerImageID string, gpus []string, modelDir string) (*Worker, error) {
	manager, err := NewDockerManager(containerImageID, gpus, modelDir)
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

	if resp.JSON200 == nil {
		slog.Error("image-to-video container returned no content")
		return nil, errors.New("image-to-video container returned no content")
	}

	return resp.JSON200, nil
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

	if resp.JSON422 != nil {
		val, err := json.Marshal(resp.JSON422)
		if err != nil {
			return nil, err
		}
		slog.Error("upscale container returned 422", slog.String("err", string(val)))
		return nil, errors.New("upscale container returned 422")
	}

	if resp.JSON400 != nil {
		val, err := json.Marshal(resp.JSON400)
		if err != nil {
			return nil, err
		}
		slog.Error("upscale container returned 400", slog.String("err", string(val)))
		return nil, errors.New("upscale container returned 400")
	}

	if resp.JSON500 != nil {
		val, err := json.Marshal(resp.JSON500)
		if err != nil {
			return nil, err
		}
		slog.Error("upscale container returned 500", slog.String("err", string(val)))
		return nil, errors.New("upscale container returned 500")
	}

	return resp.JSON200, nil
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

	if resp.JSON422 != nil {
		val, err := json.Marshal(resp.JSON422)
		if err != nil {
			return nil, err
		}
		slog.Error("audio-to-text container returned 422", slog.String("err", string(val)))
		return nil, errors.New("audio-to-text container returned 422")
	}

	if resp.JSON400 != nil {
		val, err := json.Marshal(resp.JSON400)
		if err != nil {
			return nil, err
		}
		slog.Error("audio-to-text container returned 400", slog.String("err", string(val)))
		return nil, errors.New("audio-to-text container returned 400")
	}

	if resp.JSON413 != nil {
		msg := "audio-to-text container returned 413 file too large; max file size is 50MB"
		slog.Error("audio-to-text container returned 413", slog.String("err", string(msg)))
		return nil, errors.New(msg)
	}

	if resp.JSON500 != nil {
		val, err := json.Marshal(resp.JSON500)
		if err != nil {
			return nil, err
		}
		slog.Error("audio-to-text container returned 500", slog.String("err", string(val)))
		return nil, errors.New("audio-to-text container returned 500")
	}

	return resp.JSON200, nil
}

func (w *Worker) LlmGenerate(ctx context.Context, req LlmGenerateFormdataRequestBody) (interface{}, error) {
	slog.Info("Incoming request %v", req)
	c, err := w.borrowContainer(ctx, "llm-generate", *req.ModelId)
	if err != nil {
		return nil, err
	}
	if c == nil {
		return nil, errors.New("borrowed container is nil")
	}
	if c.Client == nil {
		return nil, errors.New("container client is nil")
	}

	defer w.returnContainer(c)

	slog.Info("Container borrowed successfully", "model_id", *req.ModelId)

	var buf bytes.Buffer
	mw, err := NewLlmGenerateMultipartWriter(&buf, req)
	if err != nil {
		return nil, err
	}

	resp, err := c.Client.LlmGenerateWithBodyWithResponse(ctx, mw.FormDataContentType(), &buf)
	if err != nil {
		return nil, err
	}

	if req.Stream != nil && *req.Stream {
		return w.handleStreamingResponse(ctx, resp)
	}

	return w.handleNonStreamingResponse(resp)
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
	rc, err := NewRunnerContainer(ctx, cfg)
	if err != nil {
		return err
	}

	name := dockerContainerName(pipeline, modelID)
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
	managedCapacity := w.manager.HasCapacity(context.Background(), pipeline, modelID)
	if managedCapacity {
		return true
	}

	// Check if we have capacity for external containers.
	name := dockerContainerName(pipeline, modelID)
	w.mu.Lock()
	defer w.mu.Unlock()
	_, ok := w.externalContainers[name]

	return ok
}

func (w *Worker) borrowContainer(ctx context.Context, pipeline, modelID string) (*RunnerContainer, error) {
	w.mu.Lock()

	name := dockerContainerName(pipeline, modelID)
	rc, ok := w.externalContainers[name]
	if ok {
		w.mu.Unlock()
		// We allow concurrent in-flight requests for external containers and assume that it knows
		// how to handle them
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
		// Noop because we allow concurrent in-flight requests for external containers
	}
}

func (w *Worker) handleNonStreamingResponse(resp *LlmGenerateResponse) (*LlmResponse, error) {
	if resp.JSON400 != nil {
		val, err := json.Marshal(resp.JSON400)
		if err != nil {
			return nil, err
		}
		slog.Error("llm-generate container returned 400", slog.String("err", string(val)))
		return nil, errors.New("llm-generate container returned 400")
	}

	if resp.JSON401 != nil {
		val, err := json.Marshal(resp.JSON401)
		if err != nil {
			return nil, err
		}
		slog.Error("llm-generate container returned 401", slog.String("err", string(val)))
		return nil, errors.New("llm-generate container returned 401")
	}

	if resp.JSON500 != nil {
		val, err := json.Marshal(resp.JSON500)
		if err != nil {
			return nil, err
		}
		slog.Error("llm-generate container returned 500", slog.String("err", string(val)))
		return nil, errors.New("llm-generate container returned 500")
	}

	return resp.JSON200, nil
}

type LlmStreamChunk struct {
	Chunk      string `json:"chunk,omitempty"`
	TokensUsed int    `json:"tokens_used,omitempty"`
	Done       bool   `json:"done,omitempty"`
}

func (w *Worker) handleStreamingResponse(ctx context.Context, resp *LlmGenerateResponse) (<-chan LlmStreamChunk, error) {
	if resp.StatusCode() != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode())
	}

	outputChan := make(chan LlmStreamChunk, 10)

	go func() {
		defer close(outputChan)

		reader := bufio.NewReader(bytes.NewReader(resp.Body))
		totalTokens := 0

		for {
			select {
			case <-ctx.Done():
				return
			default:
				line, err := reader.ReadBytes('\n')
				if err != nil {
					if err != io.EOF {
						slog.Error("Error reading stream", slog.String("err", err.Error()))
					}
					return
				}

				if bytes.HasPrefix(line, []byte("data: ")) {
					data := bytes.TrimPrefix(line, []byte("data: "))
					if string(data) == "[DONE]" {
						outputChan <- LlmStreamChunk{Chunk: "[DONE]", Done: true, TokensUsed: totalTokens}
						return
					}

					var streamData LlmStreamChunk
					if err := json.Unmarshal(data, &streamData); err != nil {
						slog.Error("Error unmarshaling stream data", slog.String("err", err.Error()))
						continue
					}

					totalTokens += streamData.TokensUsed

					select {
					case outputChan <- streamData:
					case <-ctx.Done():
						return
					}
				}
			}
		}
	}()

	return outputChan, nil
}
