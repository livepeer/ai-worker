package cog

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"

	"github.com/mitchellh/go-homedir"
	"github.com/replicate/cog/pkg/docker"
	"github.com/replicate/cog/pkg/predict"
	"github.com/replicate/cog/pkg/util/mime"
	"github.com/vincent-petithory/dataurl"
)

type Config struct {
	ContainerImageID string
	GPUs             string
}

type Executor struct {
	predictor predict.Predictor
}

func NewExecutor(config Config) *Executor {
	predictor := predict.NewPredictor(docker.RunOptions{
		GPUs:    config.GPUs,
		Image:   config.ContainerImageID,
		Volumes: []docker.Volume{},
		Env:     []string{},
	})

	return &Executor{
		predictor: predictor,
	}
}

func (e *Executor) Start(ctx context.Context) error {
	// TODO: Add support for stopping on idle timeout
	go func() {
		<-ctx.Done()

		if err := e.Stop(); err != nil {
			slog.Error("Error stopping executor", slog.String("error", err.Error()))
		}
	}()

	return e.predictor.Start(os.Stderr)
}

func (e *Executor) Stop() error {
	return e.predictor.Stop()
}

// Based on https://github.com/replicate/cog/blob/main/pkg/cli/predict.go
func (e *Executor) Execute(inputs map[string]string, outputPath string) error {
	schema, err := e.predictor.GetSchema()
	if err != nil {
		return err
	}

	predInputs := predict.NewInputs(inputs)

	prediction, err := e.predictor.Predict(predInputs)
	if err != nil {
		return err
	}

	// Generate output depending on type in schema
	var out []byte
	responseSchema := schema.Paths["/predictions"].Post.Responses["200"].Value.Content["application/json"].Schema.Value
	outputSchema := responseSchema.Properties["output"].Value

	// Multiple outputs!
	// if outputSchema.Type == "array" && outputSchema.Items.Value != nil && outputSchema.Items.Value.Type == "string" && outputSchema.Items.Value.Format == "uri" {
	// 	return handleMultipleFileOutput(prediction, outputSchema)
	// }

	if outputSchema.Type == "string" && outputSchema.Format == "uri" {
		dataurlObj, err := dataurl.DecodeString((*prediction.Output).(string))
		if err != nil {
			return fmt.Errorf("Failed to decode dataurl: %w", err)
		}
		out = dataurlObj.Data
		if outputPath == "" {
			outputPath = "output"
			extension := mime.ExtensionByType(dataurlObj.ContentType())
			if extension != "" {
				outputPath += extension
			}
		}
	} else if outputSchema.Type == "string" {
		// Handle strings separately because if we encode it to JSON it will be surrounded by quotes.
		s := (*prediction.Output).(string)
		out = []byte(s)
	} else {
		// Treat everything else as JSON -- ints, floats, bools will all convert correctly.
		rawJSON, err := json.Marshal(prediction.Output)
		if err != nil {
			return fmt.Errorf("Failed to encode prediction output as JSON: %w", err)
		}
		var indentedJSON bytes.Buffer
		if err := json.Indent(&indentedJSON, rawJSON, "", "  "); err != nil {
			return err
		}
		out = indentedJSON.Bytes()

		// FIXME: this stopped working
		// f := colorjson.NewFormatter()
		// f.Indent = 2
		// s, _ := f.Marshal(obj)

	}

	return writeOutput(outputPath, out)
}

func writeOutput(outputPath string, output []byte) error {
	outputPath, err := homedir.Expand(outputPath)
	if err != nil {
		return err
	}

	// Write to file
	outFile, err := os.OpenFile(outputPath, os.O_WRONLY|os.O_CREATE, 0o755)
	if err != nil {
		return err
	}

	if _, err := outFile.Write(output); err != nil {
		return err
	}
	if err := outFile.Close(); err != nil {
		return err
	}

	return nil
}

// func handleMultipleFileOutput(prediction *predict.Response, outputSchema *openapi3.Schema) error {
// 	outputs, ok := (*prediction.Output).([]interface{})
// 	if !ok {
// 		return fmt.Errorf("Failed to decode output")
// 	}

// 	for i, output := range outputs {
// 		outputString := output.(string)
// 		dataurlObj, err := dataurl.DecodeString(outputString)
// 		if err != nil {
// 			return fmt.Errorf("Failed to decode dataurl: %w", err)
// 		}
// 		out := dataurlObj.Data
// 		extension := mime.ExtensionByType(dataurlObj.ContentType())
// 		outputPath := fmt.Sprintf("output.%d%s", i, extension)
// 		if err := writeOutput(outputPath, out); err != nil {
// 			return err
// 		}
// 	}

// 	return nil
// }
