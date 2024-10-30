// cmd/examples/inpainting/main.go

package main

import (
    "bytes"
    "fmt"
    "io"
    "log/slog"
    "mime/multipart"
    "net/http"
    "os"
    "path"
    "strconv"
)

func main() {
    if len(os.Args) < 4 {
        slog.Error("Usage: main <runs> <prompt> <image_path> <mask_path>")
        return
    }

    runs, err := strconv.Atoi(os.Args[1])
    if err != nil {
        slog.Error("Invalid runs arg", slog.String("error", err.Error()))
        return
    }

    prompt := os.Args[2]
    imagePath := os.Args[3]
    maskPath := os.Args[4]

    // Create output directory if it doesn't exist
    outputDir := "output"
    if err := os.MkdirAll(outputDir, 0755); err != nil {
        slog.Error("Error creating output directory", slog.String("error", err.Error()))
        return
    }

    // Prepare request URL
    url := "http://localhost:8600/inpainting"

    for i := 0; i < runs; i++ {
        slog.Info("Running inpainting", slog.Int("run", i+1))

        // Create multipart form data
        var b bytes.Buffer
        w := multipart.NewWriter(&b)

        // Add prompt
        if err := w.WriteField("prompt", prompt); err != nil {
            slog.Error("Error writing prompt field", slog.String("error", err.Error()))
            return
        }

        // Add image file
        imageFile, err := os.Open(imagePath)
        if err != nil {
            slog.Error("Error opening image file", slog.String("error", err.Error()))
            return
        }
        defer imageFile.Close()

        fw, err := w.CreateFormFile("image", imagePath)
        if err != nil {
            slog.Error("Error creating form file", slog.String("error", err.Error()))
            return
        }
        if _, err = io.Copy(fw, imageFile); err != nil {
            slog.Error("Error copying image file", slog.String("error", err.Error()))
            return
        }

        // Add mask file
        maskFile, err := os.Open(maskPath)
        if err != nil {
            slog.Error("Error opening mask file", slog.String("error", err.Error()))
            return
        }
        defer maskFile.Close()

        fw, err = w.CreateFormFile("mask_image", maskPath)
        if err != nil {
            slog.Error("Error creating form file", slog.String("error", err.Error()))
            return
        }
        if _, err = io.Copy(fw, maskFile); err != nil {
            slog.Error("Error copying mask file", slog.String("error", err.Error()))
            return
        }

        // Close the writer
        w.Close()

        // Create request
        req, err := http.NewRequest("POST", url, &b)
        if err != nil {
            slog.Error("Error creating request", slog.String("error", err.Error()))
            return
        }
        req.Header.Set("Content-Type", w.FormDataContentType())

        // Send request
        client := &http.Client{}
        resp, err := client.Do(req)
        if err != nil {
            slog.Error("Error sending request", slog.String("error", err.Error()))
            return
        }
        defer resp.Body.Close()

        // Check response status
        if resp.StatusCode != http.StatusOK {
            body, _ := io.ReadAll(resp.Body)
            slog.Error("Error response from server", 
                slog.Int("status", resp.StatusCode),
                slog.String("body", string(body)))
            return
        }

        // Save response to file
        outputPath := path.Join(outputDir, fmt.Sprintf("output_%d.json", i))
        out, err := os.Create(outputPath)
        if err != nil {
            slog.Error("Error creating output file", slog.String("error", err.Error()))
            return
        }
        defer out.Close()

        _, err = io.Copy(out, resp.Body)
        if err != nil {
            slog.Error("Error saving response", slog.String("error", err.Error()))
            return
        }

        slog.Info("Output written", slog.String("path", outputPath))
    }
}