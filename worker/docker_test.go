package worker

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestDockerManager_getContainerImageName(t *testing.T) {
	tests := []struct {
		name          string
		pipeline      string
		modelID       string
		expectedImage string
		expectError   bool
	}{
		{
			name:          "live-video-to-video with valid modelID",
			pipeline:      "live-video-to-video",
			modelID:       "streamdiffusion",
			expectedImage: "livepeer/ai-runner:live-app-streamdiffusion",
			expectError:   false,
		},
		{
			name:        "live-video-to-video with invalid modelID",
			pipeline:    "live-video-to-video",
			modelID:     "invalid-model",
			expectError: true,
		},
		{
			name:          "valid pipeline",
			pipeline:      "text-to-speech",
			modelID:       "",
			expectedImage: "livepeer/ai-runner:text-to-speech",
			expectError:   false,
		},
		{
			name:          "invalid pipeline",
			pipeline:      "invalid-pipeline",
			modelID:       "",
			expectedImage: "default-image",
			expectError:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			manager := &DockerManager{
				defaultImage: "default-image",
			}

			image, err := manager.getContainerImageName(tt.pipeline, tt.modelID)
			if tt.expectError {
				require.Error(t, err)
				require.Equal(t, fmt.Sprintf("no container image found for live pipeline %s", tt.modelID), err.Error())
			} else {
				require.NoError(t, err)
				require.Equal(t, tt.expectedImage, image)
			}
		})
	}
}

func TestDockerContainerName(t *testing.T) {
	tests := []struct {
		name         string
		pipeline     string
		modelID      string
		suffix       []string
		expectedName string
	}{
		{
			name:         "with suffix",
			pipeline:     "text-to-speech",
			modelID:      "model1",
			suffix:       []string{"suffix1"},
			expectedName: "text-to-speech_model1_suffix1",
		},
		{
			name:         "without suffix",
			pipeline:     "text-to-speech",
			modelID:      "model1",
			expectedName: "text-to-speech_model1",
		},
		{
			name:         "modelID with special characters",
			pipeline:     "text-to-speech",
			modelID:      "model/1_2",
			suffix:       []string{"suffix1"},
			expectedName: "text-to-speech_model-1-2_suffix1",
		},
		{
			name:         "modelID with special characters without suffix",
			pipeline:     "text-to-speech",
			modelID:      "model/1_2",
			expectedName: "text-to-speech_model-1-2",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			name := dockerContainerName(tt.pipeline, tt.modelID, tt.suffix...)
			require.Equal(t, tt.expectedName, name)
		})
	}
}
