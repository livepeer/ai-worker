// cmd/examples/inpainting/main.go

package main

import (
    "context"
    "flag"
    "log/slog"
    "os"
    "path"
    "path/filepath"
    "strconv"
    "time"

    "github.com/livepeer/ai-worker/worker"
    "github.com/oapi-codegen/runtime/types"
)

func main() {
    aiModelsDir := flag.String("aiModelsDir", "runner/models", "path to the models directory")
    flag.Parse()

    containerName := "inpainting"
    baseOutputPath := "output"

    containerImageID := "livepeer/ai-runner:inpainting"
    gpus := []string{"0"}

    modelsDir, err := filepath.Abs(*aiModelsDir)
    if err != nil {
        slog.Error("Error getting absolute path for 'aiModelsDir'", slog.String("error", err.Error()))
        return
    }

    modelID := "stabilityai/stable-diffusion-2-inpainting"

    w, err := worker.NewWorker(containerImageID, gpus, modelsDir)
    if err != nil {
        slog.Error("Error creating worker", slog.String("error", err.Error()))
        return
    }

    slog.Info("Warming container")

    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    endpoint := worker.RunnerEndpoint{
        Port: "8600",
    }

    if err := w.Warm(ctx, containerName, modelID, endpoint, worker.OptimizationFlags{}); err != nil {
        slog.Error("Error warming container", slog.String("error", err.Error()))
        return
    }

    slog.Info("Warm container is up")

    args := os.Args[1:]
    if len(args) < 4 {
        slog.Error("Usage: main <runs> <prompt> <image_path> <mask_path>")
        return
    }

    runs, err := strconv.Atoi(args[0])
    if err != nil {
        slog.Error("Invalid runs arg", slog.String("error", err.Error()))
        return
    }

    prompt := args[1]
    imagePath := args[2]
    maskPath := args[3]

    // Read and prepare image file
    imageBytes, err := os.ReadFile(imagePath)
    if err != nil {
        slog.Error("Error reading image", slog.String("imagePath", imagePath))
        return
    }
    imageFile := types.File{}
    imageFile.InitFromBytes(imageBytes, imagePath)

    // Read and prepare mask file
    maskBytes, err := os.ReadFile(maskPath)
    if err != nil {
        slog.Error("Error reading mask", slog.String("maskPath", maskPath))
        return
    }
    maskFile := types.File{}
    maskFile.InitFromBytes(maskBytes, maskPath)

    // Prepare request
    req := worker.GenInpaintingMultipartRequestBody{
        Image:     imageFile,
        MaskImage: maskFile,
        ModelId:   &modelID,
        Prompt:    prompt,
        Strength:  worker.Float32(1.0),
        GuidanceScale: worker.Float32(7.5),
        NumInferenceSteps: worker.Int32(50),
        SafetyCheck: worker.Bool(true),
    }

    // Create output directory if it doesn't exist
    if err := os.MkdirAll(baseOutputPath, 0755); err != nil {
        slog.Error("Error creating output directory", slog.String("error", err.Error()))
        return
    }

    for i := 0; i < runs; i++ {
        slog.Info("Running inpainting", slog.Int("num", i))

        resp, err := w.Inpainting(ctx, req)
        if err != nil {
            slog.Error("Error running inpainting", slog.String("error", err.Error()))
            return
        }

        for j, media := range resp.Images {
            outputPath := path.Join(baseOutputPath, strconv.Itoa(i)+"_"+strconv.Itoa(j)+".png")
            if err := worker.SaveImageB64DataUrl(media.Url, outputPath); err != nil {
                slog.Error("Error saving b64 data url as image", slog.String("error", err.Error()))
                return
            }

            slog.Info("Output written", 
                slog.String("outputPath", outputPath),
                slog.Int("seed", media.Seed),
                slog.Bool("nsfw", media.Nsfw),
            )
        }
    }

    slog.Info("Sleeping 2 seconds and then stopping container")
    time.Sleep(2 * time.Second)
    w.Stop(ctx)
    time.Sleep(1 * time.Second)
}