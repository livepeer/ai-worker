package worker

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"testing"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/image"
	"github.com/docker/docker/api/types/network"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

type MockDockerClient struct {
	mock.Mock
}

func (m *MockDockerClient) ImagePull(ctx context.Context, ref string, options image.PullOptions) (io.ReadCloser, error) {
	args := m.Called(ctx, ref, options)
	return args.Get(0).(io.ReadCloser), args.Error(1)
}

func (m *MockDockerClient) ImageInspectWithRaw(ctx context.Context, imageID string) (types.ImageInspect, []byte, error) {
	args := m.Called(ctx, imageID)
	return args.Get(0).(types.ImageInspect), args.Get(1).([]byte), args.Error(2)
}

func (m *MockDockerClient) ContainerCreate(ctx context.Context, config *container.Config, hostConfig *container.HostConfig, networkingConfig *network.NetworkingConfig, platform *ocispec.Platform, containerName string) (container.CreateResponse, error) {
	args := m.Called(ctx, config, hostConfig, networkingConfig, platform, containerName)
	return args.Get(0).(container.CreateResponse), args.Error(1)
}

func (m *MockDockerClient) ContainerStart(ctx context.Context, containerID string, options container.StartOptions) error {
	args := m.Called(ctx, containerID, options)
	return args.Error(0)
}

func (m *MockDockerClient) ContainerInspect(ctx context.Context, containerID string) (types.ContainerJSON, error) {
	args := m.Called(ctx, containerID)
	return args.Get(0).(types.ContainerJSON), args.Error(1)
}

func (m *MockDockerClient) ContainerList(ctx context.Context, options container.ListOptions) ([]types.Container, error) {
	args := m.Called(ctx, options)
	return args.Get(0).([]types.Container), args.Error(1)
}

func (m *MockDockerClient) ContainerStop(ctx context.Context, containerID string, options container.StopOptions) error {
	args := m.Called(ctx, containerID, options)
	return args.Error(0)
}

func (m *MockDockerClient) ContainerRemove(ctx context.Context, containerID string, options container.RemoveOptions) error {
	args := m.Called(ctx, containerID, options)
	return args.Error(0)
}

func TestGetContainerImage(t *testing.T) {
	dockerManager := &DockerManager{
		defaultImage: "default-image",
	}

	tests := []struct {
		pipeline string
		modelID  string
		expected string
		err      bool
	}{
		{"segment-anything-2", "", "livepeer/ai-runner:segment-anything-2", false},
		{"text-to-speech", "", "livepeer/ai-runner:text-to-speech", false},
		{"live-video-to-video", "streamdiffusion", "livepeer/ai-runner:live-app-streamdiffusion", false},
		{"live-video-to-video", "unknown-model", "", true},
		{"unknown-pipeline", "", "default-image", false},
	}

	for _, tt := range tests {
		image, err := dockerManager.getContainerImageName(tt.pipeline, tt.modelID)
		if tt.err {
			require.Error(t, err)
		} else {
			require.NoError(t, err)
			require.Equal(t, tt.expected, image)
		}
	}
}

func TestPullImageAsync(t *testing.T) {
	mockDockerClient := new(MockDockerClient)

	// Mock the ContainerList method to simulate the removal of existing containers.
	mockDockerClient.On("ContainerList", mock.Anything, mock.Anything).Return([]types.Container{}, nil)

	dockerManager, err := NewDockerManager("default-image", []string{"gpu0"}, "/models", mockDockerClient)
	require.NoError(t, err)

	ctx := context.Background()
	imageName := "test-image"

	// Mock the ImageInspectWithRaw method to simulate the image being available locally.
	mockDockerClient.On("ImageInspectWithRaw", mock.Anything, imageName).Return(types.ImageInspect{}, []byte{}, nil)

	// Call pullImageAsync and verify that it does not attempt to pull the image.
	dockerManager.pullImageAsync(ctx, imageName)
	mockDockerClient.AssertNotCalled(t, "ImagePull", mock.Anything, imageName, mock.Anything)

	// Mock the ImageInspectWithRaw method to simulate the image not being available locally.
	mockDockerClient.On("ImageInspectWithRaw", mock.Anything, imageName).Return(types.ImageInspect{}, []byte{}, errors.New("not found"))

	// Mock the ImagePull method to simulate pulling the image.
	mockDockerClient.On("ImagePull", mock.Anything, imageName, mock.Anything).Return(io.NopCloser(strings.NewReader("")), nil)

	// Call pullImageAsync and verify that it attempts to pull the image.
	dockerManager.pullImageAsync(ctx, imageName)
	mockDockerClient.AssertCalled(t, "ImagePull", mock.Anything, imageName, mock.Anything)
}

func TestPullImage(t *testing.T) {
	mockDockerClient := new(MockDockerClient)

	// Mock the ContainerList method to simulate the removal of existing containers.
	mockDockerClient.On("ContainerList", mock.Anything, mock.Anything).Return([]types.Container{}, nil)

	dockerManager, err := NewDockerManager("default-image", []string{"gpu0"}, "/models", mockDockerClient)
	require.NoError(t, err)

	ctx := context.Background()
	imageName := "test-image"

	// Mock the ImagePull method to simulate pulling the image.
	mockDockerClient.On("ImagePull", mock.Anything, imageName, mock.Anything).Return(io.NopCloser(strings.NewReader("")), nil)

	// Call pullImage and verify that it pulls the image.
	err = dockerManager.pullImage(ctx, imageName)
	require.NoError(t, err)
	mockDockerClient.AssertCalled(t, "ImagePull", mock.Anything, imageName, mock.Anything)
}

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
