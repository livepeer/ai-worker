package worker

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
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

func (w *Worker) TextToImage(ctx context.Context, req GenTextToImageJSONRequestBody) (*ImageResponse, error) {
	c, err := w.borrowContainer(ctx, "text-to-image", *req.ModelId)
	if err != nil {
		return nil, err
	}
	defer w.returnContainer(c)

	resp, err := c.Client.GenTextToImageWithResponse(ctx, req)
	if err != nil {
		return nil, err
	}

	if resp.JSON400 != nil {
		val, err := json.Marshal(resp.JSON400)
		if err != nil {
			return nil, err
		}
		slog.Error("text-to-image container returned 400", slog.String("err", string(val)))
		return nil, errors.New("text-to-image container returned 400: " + resp.JSON400.Detail.Msg)
	}

	if resp.JSON401 != nil {
		val, err := json.Marshal(resp.JSON401)
		if err != nil {
			return nil, err
		}
		slog.Error("text-to-image container returned 401", slog.String("err", string(val)))
		return nil, errors.New("text-to-image container returned 401: " + resp.JSON401.Detail.Msg)
	}

	if resp.JSON422 != nil {
		val, err := json.Marshal(resp.JSON422)
		if err != nil {
			return nil, err
		}
		slog.Error("text-to-image container returned 422", slog.String("err", string(val)))
		return nil, errors.New("text-to-image container returned 422: " + string(val))
	}

	if resp.JSON500 != nil {
		val, err := json.Marshal(resp.JSON500)
		if err != nil {
			return nil, err
		}
		slog.Error("text-to-image container returned 500", slog.String("err", string(val)))
		return nil, errors.New("text-to-image container returned 500: " + resp.JSON500.Detail.Msg)
	}

	return resp.JSON200, nil
}

func (w *Worker) ImageToImage(ctx context.Context, req GenImageToImageMultipartRequestBody) (*ImageResponse, error) {
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

	resp, err := c.Client.GenImageToImageWithBodyWithResponse(ctx, mw.FormDataContentType(), &buf)
	if err != nil {
		return nil, err
	}

	if resp.JSON400 != nil {
		val, err := json.Marshal(resp.JSON400)
		if err != nil {
			return nil, err
		}
		slog.Error("image-to-image container returned 400", slog.String("err", string(val)))
		return nil, errors.New("image-to-image container returned 400: " + resp.JSON400.Detail.Msg)
	}

	if resp.JSON401 != nil {
		val, err := json.Marshal(resp.JSON401)
		if err != nil {
			return nil, err
		}
		slog.Error("image-to-image container returned 401", slog.String("err", string(val)))
		return nil, errors.New("image-to-image container returned 401: " + resp.JSON401.Detail.Msg)
	}

	if resp.JSON422 != nil {
		val, err := json.Marshal(resp.JSON422)
		if err != nil {
			return nil, err
		}
		slog.Error("image-to-image container returned 422", slog.String("err", string(val)))
		return nil, errors.New("image-to-image container returned 422: " + string(val))
	}

	if resp.JSON500 != nil {
		val, err := json.Marshal(resp.JSON500)
		if err != nil {
			return nil, err
		}
		slog.Error("image-to-image container returned 500", slog.String("err", string(val)))
		return nil, errors.New("image-to-image container returned 500: " + resp.JSON500.Detail.Msg)
	}

	return resp.JSON200, nil
}

func (w *Worker) ImageToVideo(ctx context.Context, req GenImageToVideoMultipartRequestBody) (*VideoResponse, error) {
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

	resp, err := c.Client.GenImageToVideoWithBodyWithResponse(ctx, mw.FormDataContentType(), &buf)
	if err != nil {
		return nil, err
	}

	if resp.JSON400 != nil {
		val, err := json.Marshal(resp.JSON400)
		if err != nil {
			return nil, err
		}
		slog.Error("image-to-video container returned 400", slog.String("err", string(val)))
		return nil, errors.New("image-to-video container returned 400: " + resp.JSON400.Detail.Msg)
	}

	if resp.JSON401 != nil {
		val, err := json.Marshal(resp.JSON401)
		if err != nil {
			return nil, err
		}
		slog.Error("image-to-video container returned 401", slog.String("err", string(val)))
		return nil, errors.New("image-to-video container returned 401: " + resp.JSON401.Detail.Msg)
	}

	if resp.JSON422 != nil {
		val, err := json.Marshal(resp.JSON422)
		if err != nil {
			return nil, err
		}
		slog.Error("image-to-video container returned 422", slog.String("err", string(val)))
		return nil, errors.New("image-to-video container returned 422: " + string(val))
	}

	if resp.JSON500 != nil {
		val, err := json.Marshal(resp.JSON500)
		if err != nil {
			return nil, err
		}
		slog.Error("image-to-video container returned 500", slog.String("err", string(val)))
		return nil, errors.New("image-to-video container returned 500: " + resp.JSON500.Detail.Msg)
	}

	if resp.JSON200 == nil {
		slog.Error("image-to-video container returned no content")
		return nil, errors.New("image-to-video container returned no content")
	}

	return resp.JSON200, nil
}

func (w *Worker) Upscale(ctx context.Context, req GenUpscaleMultipartRequestBody) (*ImageResponse, error) {
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

	resp, err := c.Client.GenUpscaleWithBodyWithResponse(ctx, mw.FormDataContentType(), &buf)
	if err != nil {
		return nil, err
	}

	if resp.JSON400 != nil {
		val, err := json.Marshal(resp.JSON400)
		if err != nil {
			return nil, err
		}
		slog.Error("upscale container returned 400", slog.String("err", string(val)))
		return nil, errors.New("upscale container returned 400: " + resp.JSON400.Detail.Msg)
	}

	if resp.JSON401 != nil {
		val, err := json.Marshal(resp.JSON401)
		if err != nil {
			return nil, err
		}
		slog.Error("upscale container returned 401", slog.String("err", string(val)))
		return nil, errors.New("upscale container returned 401: " + resp.JSON401.Detail.Msg)
	}

	if resp.JSON422 != nil {
		val, err := json.Marshal(resp.JSON422)
		if err != nil {
			return nil, err
		}
		slog.Error("upscale container returned 422", slog.String("err", string(val)))
		return nil, errors.New("upscale container returned 422: " + string(val))
	}

	if resp.JSON500 != nil {
		val, err := json.Marshal(resp.JSON500)
		if err != nil {
			return nil, err
		}
		slog.Error("upscale container returned 500", slog.String("err", string(val)))
		return nil, errors.New("upscale container returned 500: " + resp.JSON500.Detail.Msg)
	}

	return resp.JSON200, nil
}

func (w *Worker) AudioToText(ctx context.Context, req GenAudioToTextMultipartRequestBody) (*TextResponse, error) {
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

	resp, err := c.Client.GenAudioToTextWithBodyWithResponse(ctx, mw.FormDataContentType(), &buf)
	if err != nil {
		return nil, err
	}

	if resp.JSON400 != nil {
		val, err := json.Marshal(resp.JSON400)
		if err != nil {
			return nil, err
		}
		slog.Error("audio-to-text container returned 400", slog.String("err", string(val)))
		return nil, errors.New("audio-to-text container returned 400: " + resp.JSON400.Detail.Msg)
	}

	if resp.JSON401 != nil {
		val, err := json.Marshal(resp.JSON401)
		if err != nil {
			return nil, err
		}
		slog.Error("audio-to-text container returned 401", slog.String("err", string(val)))
		return nil, errors.New("audio-to-text container returned 401: " + resp.JSON401.Detail.Msg)
	}

	if resp.JSON413 != nil {
		msg := "audio-to-text container returned 413: file too large; max file size is 50MB"
		slog.Error("audio-to-text container returned 413", slog.String("err", string(msg)))
		return nil, errors.New(msg)
	}

	if resp.JSON415 != nil {
		val, err := json.Marshal(resp.JSON415)
		if err != nil {
			return nil, err
		}
		slog.Error("audio-to-text container returned 415", slog.String("err", string(val)))
		return nil, errors.New("audio-to-text container returned 415: " + resp.JSON415.Detail.Msg)
	}

	if resp.JSON422 != nil {
		val, err := json.Marshal(resp.JSON422)
		if err != nil {
			return nil, err
		}
		slog.Error("audio-to-text container returned 422", slog.String("err", string(val)))
		return nil, errors.New("audio-to-text container returned 422: " + string(val))
	}

	if resp.JSON500 != nil {
		val, err := json.Marshal(resp.JSON500)
		if err != nil {
			return nil, err
		}
		slog.Error("audio-to-text container returned 500", slog.String("err", string(val)))
		return nil, errors.New("audio-to-text container returned 500: " + resp.JSON500.Detail.Msg)
	}

	return resp.JSON200, nil
}

func (w *Worker) LLM(ctx context.Context, req GenLLMFormdataRequestBody) (interface{}, error) {
	c, err := w.borrowContainer(ctx, "llm", *req.ModelId)
	if err != nil {
		return nil, err
	}
	if c == nil {
		return nil, errors.New("borrowed container is nil")
	}
	if c.Client == nil {
		return nil, errors.New("container client is nil")
	}

	slog.Info("Container borrowed successfully", "model_id", *req.ModelId)

	var buf bytes.Buffer
	mw, err := NewLLMMultipartWriter(&buf, req)
	if err != nil {
		return nil, err
	}

	if req.Stream != nil && *req.Stream {
		resp, err := c.Client.GenLLMWithBody(ctx, mw.FormDataContentType(), &buf)
		if err != nil {
			return nil, err
		}
		return w.handleStreamingResponse(ctx, c, resp)
	}

	resp, err := c.Client.GenLLMWithBodyWithResponse(ctx, mw.FormDataContentType(), &buf)
	if err != nil {
		return nil, err
	}
	return w.handleNonStreamingResponse(c, resp)
}

func (w *Worker) SegmentAnything2(ctx context.Context, req GenSegmentAnything2MultipartRequestBody) (*MasksResponse, error) {
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

	resp, err := c.Client.GenSegmentAnything2WithBodyWithResponse(ctx, mw.FormDataContentType(), &buf)
	if err != nil {
		return nil, err
	}

	if resp.JSON400 != nil {
		val, err := json.Marshal(resp.JSON400)
		if err != nil {
			return nil, err
		}
		slog.Error("segment anything 2 container returned 400", slog.String("err", string(val)))
		return nil, errors.New("segment anything 2 container returned 400: " + resp.JSON400.Detail.Msg)
	}

	if resp.JSON401 != nil {
		val, err := json.Marshal(resp.JSON401)
		if err != nil {
			return nil, err
		}
		slog.Error("segment anything 2 container returned 401", slog.String("err", string(val)))
		return nil, errors.New("segment anything 2 container returned 401: " + resp.JSON401.Detail.Msg)
	}

	if resp.JSON422 != nil {
		val, err := json.Marshal(resp.JSON422)
		if err != nil {
			return nil, err
		}
		slog.Error("segment anything 2 container returned 422", slog.String("err", string(val)))
		return nil, errors.New("segment anything 2 container returned 422: " + string(val))
	}

	if resp.JSON500 != nil {
		val, err := json.Marshal(resp.JSON500)
		if err != nil {
			return nil, err
		}
		slog.Error("segment anything 2 container returned 500", slog.String("err", string(val)))
		return nil, errors.New("segment anything 2 container returned 500: " + resp.JSON500.Detail.Msg)
	}

	return resp.JSON200, nil
}

func (w *Worker) ImageToText(ctx context.Context, req GenImageToTextMultipartRequestBody) (*ImageToTextResponse, error) {
	c, err := w.borrowContainer(ctx, "image-to-text", *req.ModelId)
	if err != nil {
		return nil, err
	}
	defer w.returnContainer(c)

	var buf bytes.Buffer
	mw, err := NewImageToTextMultipartWriter(&buf, req)
	if err != nil {
		return nil, err
	}

	resp, err := c.Client.GenImageToTextWithBodyWithResponse(ctx, mw.FormDataContentType(), &buf)
	if err != nil {
		return nil, err
	}

	if resp.JSON400 != nil {
		val, err := json.Marshal(resp.JSON400)
		if err != nil {
			return nil, err
		}
		slog.Error("image-to-text container returned 400", slog.String("err", string(val)))
		return nil, errors.New("image-to-text container returned 400: " + resp.JSON400.Detail.Msg)
	}

	if resp.JSON401 != nil {
		val, err := json.Marshal(resp.JSON401)
		if err != nil {
			return nil, err
		}
		slog.Error("image-to-text container returned 401", slog.String("err", string(val)))
		return nil, errors.New("image-to-text container returned 401: " + resp.JSON401.Detail.Msg)
	}

	if resp.JSON413 != nil {
		msg := "image-to-text container returned 413 file too large; max file size is 50MB"
		slog.Error("image-to-text container returned 413", slog.String("err", string(msg)))
		return nil, errors.New(msg)
	}

	if resp.JSON422 != nil {
		val, err := json.Marshal(resp.JSON422)
		if err != nil {
			return nil, err
		}
		slog.Error("image-to-text container returned 422", slog.String("err", string(val)))
		return nil, errors.New("image-to-text  container returned 422: " + string(val))
	}

	if resp.JSON500 != nil {
		val, err := json.Marshal(resp.JSON500)
		if err != nil {
			return nil, err
		}
		slog.Error("image-to-text container returned 500", slog.String("err", string(val)))
		return nil, errors.New("image-to-text container returned 500: " + resp.JSON500.Detail.Msg)
	}

	return resp.JSON200, nil
}

func (w *Worker) SketchToImage(ctx context.Context, req GenSketchToImageMultipartRequestBody) (*ImageResponse, error) {
	c, err := w.borrowContainer(ctx, "sketch-to-image", *req.ModelId)
	if err != nil {
		return nil, err
	}
	defer w.returnContainer(c)

	var buf bytes.Buffer
	mw, err := NewSketchToImageMultipartWriter(&buf, req)
	if err != nil {
		return nil, err
	}

	resp, err := c.Client.GenSketchToImageWithBodyWithResponse(ctx, mw.FormDataContentType(), &buf)
	if err != nil {
		return nil, err
	}

	if resp.JSON422 != nil {
		val, err := json.Marshal(resp.JSON422)
		if err != nil {
			return nil, err
		}
		slog.Error("sketch-to-image container returned 422", slog.String("err", string(val)))
		return nil, errors.New("sketch-to-image container returned 422")
	}

	if resp.JSON400 != nil {
		val, err := json.Marshal(resp.JSON400)
		if err != nil {
			return nil, err
		}
		slog.Error("sketch-to-image container returned 400", slog.String("err", string(val)))
		return nil, errors.New("sketch-to-image container returned 400: " + resp.JSON400.Detail.Msg)
	}

	if resp.JSON500 != nil {
		val, err := json.Marshal(resp.JSON500)
		if err != nil {
			return nil, err
		}
		slog.Error("sketch-to-image container returned 500", slog.String("err", string(val)))
		return nil, errors.New("sketch-to-image container returned 500")
	}

	return resp.JSON200, nil
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

func (w *Worker) handleNonStreamingResponse(c *RunnerContainer, resp *GenLLMResponse) (*LLMResponse, error) {
	defer w.returnContainer(c)
	if resp.JSON400 != nil {
		val, err := json.Marshal(resp.JSON400)
		if err != nil {
			return nil, err
		}
		slog.Error("LLM container returned 400", slog.String("err", string(val)))
		return nil, errors.New("LLM container returned 400: " + resp.JSON400.Detail.Msg)
	}

	if resp.JSON401 != nil {
		val, err := json.Marshal(resp.JSON401)
		if err != nil {
			return nil, err
		}
		slog.Error("LLM container returned 401", slog.String("err", string(val)))
		return nil, errors.New("LLM container returned 401: " + resp.JSON401.Detail.Msg)
	}

	if resp.JSON500 != nil {
		val, err := json.Marshal(resp.JSON500)
		if err != nil {
			return nil, err
		}
		slog.Error("LLM container returned 500", slog.String("err", string(val)))
		return nil, errors.New("LLM container returned 500: " + resp.JSON500.Detail.Msg)
	}

	return resp.JSON200, nil
}

type LlmStreamChunk struct {
	Chunk      string `json:"chunk,omitempty"`
	TokensUsed int    `json:"tokens_used,omitempty"`
	Done       bool   `json:"done,omitempty"`
}

func (w *Worker) handleStreamingResponse(ctx context.Context, c *RunnerContainer, resp *http.Response) (<-chan LlmStreamChunk, error) {
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	outputChan := make(chan LlmStreamChunk, 10)

	go func() {
		defer close(outputChan)
		defer w.returnContainer(c)

		scanner := bufio.NewScanner(resp.Body)
		totalTokens := 0

		for scanner.Scan() {
			select {
			case <-ctx.Done():
				return
			default:
				line := scanner.Text()
				if strings.HasPrefix(line, "data: ") {
					data := strings.TrimPrefix(line, "data: ")
					if data == "[DONE]" {
						outputChan <- LlmStreamChunk{Chunk: "[DONE]", Done: true, TokensUsed: totalTokens}
						return
					}

					var streamData LlmStreamChunk
					if err := json.Unmarshal([]byte(data), &streamData); err != nil {
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

		if err := scanner.Err(); err != nil {
			slog.Error("Error reading stream", slog.String("err", err.Error()))
		}
	}()

	return outputChan, nil
}
