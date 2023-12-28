package worker

import (
	"context"
	"fmt"
	"path"
	"strconv"
)

type GenerateImageParams struct {
	Prompt            string  `json:"prompt"`
	GuidanceScale     float64 `json:"guidance_scale"`
	NumInferenceSteps int64   `json:"num_inference_steps"`
}

type GenerateImageRequest struct {
	Params           GenerateImageParams
	JobID            string
	ContainerImageID string
}

type GenerateVideoParams struct {
	Image            string  `json:"image"`
	MotionBucketID   float64 `json:"motion_bucket_id"`
	NoiseAugStrength float64 `json:"noise_aug_strength"`
}

type GenerateVideoRequest struct {
	Params           GenerateVideoParams
	JobID            string
	ContainerImageID string
}

type Executor interface {
	Start(context.Context) error
	Stop() error
	Execute(map[string]string, string) error
}

type ExecutorConfig struct {
	ContainerImageID string
	GPUs             string
}

type NewExecutorFn func(config ExecutorConfig) Executor

type Worker struct {
	gpus          string
	outputDir     string
	newExecutorFn NewExecutorFn

	executors map[string]Executor
}

func NewWorker(gpus string, outputDir string, newExecutorFn NewExecutorFn) *Worker {
	return &Worker{
		gpus:          gpus,
		outputDir:     outputDir,
		newExecutorFn: newExecutorFn,
		executors:     make(map[string]Executor),
	}
}

func (w *Worker) GenerateImage(ctx context.Context, req GenerateImageRequest) (string, error) {
	exec, err := w.getWarmExecutor(ctx, req.ContainerImageID)
	if err != nil {
		return "", err
	}

	inputs := map[string]string{
		"prompt":              req.Params.Prompt,
		"guidance_scale":      fmt.Sprintf("%f", req.Params.GuidanceScale),
		"num_inference_steps": strconv.FormatInt(req.Params.NumInferenceSteps, 10),
	}
	outputPath := path.Join(w.outputDir, req.JobID+"_out.png")
	if err := exec.Execute(inputs, outputPath); err != nil {
		return "", err
	}

	return outputPath, nil
}

func (w *Worker) GenerateVideo(ctx context.Context, req GenerateVideoRequest) (string, error) {
	exec, err := w.getWarmExecutor(ctx, req.ContainerImageID)
	if err != nil {
		return "", err
	}

	inputs := map[string]string{
		"image":              req.Params.Image,
		"motion_bucket_id":   fmt.Sprintf("%f", req.Params.MotionBucketID),
		"noise_aug_strength": fmt.Sprintf("%f", req.Params.NoiseAugStrength),
	}
	outputPath := path.Join(w.outputDir, req.JobID+"_out.mp4")
	if err := exec.Execute(inputs, outputPath); err != nil {
		return "", err
	}

	return outputPath, nil
}

func (w *Worker) Warm(ctx context.Context, containerImageID string) error {
	_, err := w.getWarmExecutor(ctx, containerImageID)
	return err
}

// Returns a warm executor for the containerImageID and creates one if it does not exist.
func (w *Worker) getWarmExecutor(ctx context.Context, containerImageID string) (Executor, error) {
	exec, ok := w.executors[containerImageID]
	if !ok {
		exec = w.newExecutorFn(ExecutorConfig{
			ContainerImageID: containerImageID,
			GPUs:             w.gpus,
		})
		w.executors[containerImageID] = exec

		// Ensures that:
		// - The image exists (i.e. was pulled from a registry)
		// - The container is up
		// - The models have been pre-loaded
		if err := exec.Start(ctx); err != nil {
			return nil, err
		}
	}

	return exec, nil
}
