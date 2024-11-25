package worker

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"testing"
	"time"

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

func createDockerManager(mockDockerClient *MockDockerClient) *DockerManager {
	return &DockerManager{
		defaultImage:    "default-image",
		gpus:            []string{"gpu0"},
		modelDir:        "/models",
		dockerClient:    mockDockerClient,
		gpuContainers:   make(map[string]string),
		containers:      make(map[string]*RunnerContainer),
		imagePullStatus: &sync.Map{},
		mu:              &sync.Mutex{},
	}
}

func TestNewDockerManager(t *testing.T) {
	mockDockerClient := new(MockDockerClient)

	createAndVerifyManager := func() *DockerManager {
		manager, err := NewDockerManager("default-image", []string{"gpu0"}, "/models", mockDockerClient)
		require.NoError(t, err)
		require.NotNil(t, manager)
		require.Equal(t, "default-image", manager.defaultImage)
		require.Equal(t, []string{"gpu0"}, manager.gpus)
		require.Equal(t, "/models", manager.modelDir)
		require.Equal(t, mockDockerClient, manager.dockerClient)
		return manager
	}

	t.Run("NoExistingContainers", func(t *testing.T) {
		mockDockerClient.On("ContainerList", mock.Anything, mock.Anything).Return([]types.Container{}, nil).Once()
		createAndVerifyManager()
		mockDockerClient.AssertNotCalled(t, "ContainerStop", mock.Anything, mock.Anything, mock.Anything)
		mockDockerClient.AssertNotCalled(t, "ContainerRemove", mock.Anything, mock.Anything, mock.Anything)
		mockDockerClient.AssertExpectations(t)
	})

	t.Run("ExistingContainers", func(t *testing.T) {
		// Mock client methods to simulate the removal of existing containers.
		existingContainers := []types.Container{
			{ID: "container1", Names: []string{"/container1"}},
			{ID: "container2", Names: []string{"/container2"}},
		}
		mockDockerClient.On("ContainerList", mock.Anything, mock.Anything).Return(existingContainers, nil)
		mockDockerClient.On("ContainerStop", mock.Anything, "container1", mock.Anything).Return(nil)
		mockDockerClient.On("ContainerStop", mock.Anything, "container2", mock.Anything).Return(nil)
		mockDockerClient.On("ContainerRemove", mock.Anything, "container1", mock.Anything).Return(nil)
		mockDockerClient.On("ContainerRemove", mock.Anything, "container2", mock.Anything).Return(nil)

		// Verify that existing containers were stopped and removed.
		createAndVerifyManager()
		mockDockerClient.AssertCalled(t, "ContainerStop", mock.Anything, "container1", mock.Anything)
		mockDockerClient.AssertCalled(t, "ContainerStop", mock.Anything, "container2", mock.Anything)
		mockDockerClient.AssertCalled(t, "ContainerRemove", mock.Anything, "container1", mock.Anything)
		mockDockerClient.AssertCalled(t, "ContainerRemove", mock.Anything, "container2", mock.Anything)
		mockDockerClient.AssertExpectations(t)
	})
}

func TestDockerManager_Warm(t *testing.T) {
	ctx := context.Background()
	pipeline := "text-to-image"
	modelID := "test-model"
	optimizationFlags := OptimizationFlags{}

	t.Run("ImageNotAvailable", func(t *testing.T) {
		mockDockerClient := new(MockDockerClient)
		dockerManager := createDockerManager(mockDockerClient)

		// Mock client methods to simulate the pulling of the image and creation of the container.
		mockDockerClient.On("ImageInspectWithRaw", mock.Anything, "default-image").Return(types.ImageInspect{}, []byte{}, errors.New("not found"))
		mockDockerClient.On("ImagePull", mock.Anything, "default-image", mock.Anything).Return(io.NopCloser(strings.NewReader("")), nil)
		mockDockerClient.On("ContainerCreate", mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(container.CreateResponse{ID: "container1"}, nil)
		mockDockerClient.On("ContainerStart", mock.Anything, "container1", mock.Anything).Return(nil)

		originalFunc := dockerWaitUntilRunningFunc
		dockerWaitUntilRunningFunc = func(ctx context.Context, client DockerClient, containerID string, pollingInterval time.Duration) error {
			return nil
		}
		defer func() { dockerWaitUntilRunningFunc = originalFunc }()

		originalFunc2 := runnerWaitUntilReadyFunc
		runnerWaitUntilReadyFunc = func(ctx context.Context, client *ClientWithResponses, pollingInterval time.Duration) error {
			return nil
		}
		defer func() { runnerWaitUntilReadyFunc = originalFunc2 }()

		err := dockerManager.Warm(ctx, pipeline, modelID, optimizationFlags)
		require.NoError(t, err)

		// Verify that the image was pulled and the container was created and started.
		mockDockerClient.AssertCalled(t, "ImagePull", mock.Anything, "default-image", mock.Anything)
		mockDockerClient.AssertCalled(t, "ContainerCreate", mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything)
		mockDockerClient.AssertCalled(t, "ContainerStart", mock.Anything, "container1", mock.Anything)
		mockDockerClient.AssertExpectations(t)
	})

	t.Run("ImageAvailable", func(t *testing.T) {
		mockDockerClient := new(MockDockerClient)
		dockerManager := createDockerManager(mockDockerClient)

		// Mock client methods to simulate the image being available locally and creation of the container.
		mockDockerClient.On("ImageInspectWithRaw", mock.Anything, "default-image").Return(types.ImageInspect{}, []byte{}, nil)
		mockDockerClient.On("ContainerCreate", mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(container.CreateResponse{ID: "container1"}, nil)
		mockDockerClient.On("ContainerStart", mock.Anything, "container1", mock.Anything).Return(nil)

		originalFunc := dockerWaitUntilRunningFunc
		dockerWaitUntilRunningFunc = func(ctx context.Context, client DockerClient, containerID string, pollingInterval time.Duration) error {
			return nil
		}
		defer func() { dockerWaitUntilRunningFunc = originalFunc }()

		originalFunc2 := runnerWaitUntilReadyFunc
		runnerWaitUntilReadyFunc = func(ctx context.Context, client *ClientWithResponses, pollingInterval time.Duration) error {
			return nil
		}
		defer func() { runnerWaitUntilReadyFunc = originalFunc2 }()

		err := dockerManager.Warm(ctx, pipeline, modelID, optimizationFlags)
		require.NoError(t, err)

		// Verify that the image was not pulled and the container was created and started.
		mockDockerClient.AssertNotCalled(t, "ImagePull", mock.Anything, "default-image", mock.Anything)
		mockDockerClient.AssertCalled(t, "ContainerCreate", mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything)
		mockDockerClient.AssertCalled(t, "ContainerStart", mock.Anything, "container1", mock.Anything)
		mockDockerClient.AssertExpectations(t)
	})
}

func TestGetContainerImage(t *testing.T) {
	mockDockerClient := new(MockDockerClient)
	dockerManager := createDockerManager(mockDockerClient)

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

func TestDockerManager_HasCapacity(t *testing.T) {
	ctx := context.Background()
	pipeline := "text-to-image"
	modelID := "test-model"

	tests := []struct {
		name                string
		setup               func(*DockerManager, *MockDockerClient)
		expectedHasCapacity bool
	}{
		{
			name: "UnusedManagedContainerExists",
			setup: func(dockerManager *DockerManager, mockDockerClient *MockDockerClient) {
				// Add an unused managed container
				dockerManager.containers["container1"] = &RunnerContainer{
					RunnerContainerConfig: RunnerContainerConfig{
						Pipeline: pipeline,
						ModelID:  modelID,
					}}
			},
			expectedHasCapacity: true,
		},
		{
			name: "ImageNotAvailable",
			setup: func(dockerManager *DockerManager, mockDockerClient *MockDockerClient) {
				// Mock client methods to simulate the image not being available locally.
				mockDockerClient.On("ImageInspectWithRaw", mock.Anything, "default-image").Return(types.ImageInspect{}, []byte{}, errors.New("not found"))
			},
			expectedHasCapacity: false,
		},
		{
			name: "ImageBeingPulled",
			setup: func(dockerManager *DockerManager, mockDockerClient *MockDockerClient) {
				// Mock client methods to simulate the image not being available locally.
				mockDockerClient.On("ImageInspectWithRaw", mock.Anything, "default-image").Return(types.ImageInspect{}, []byte{}, errors.New("not found"))
				// Mark the image as being pulled.
				dockerManager.imagePullStatus.Store("default-image", true)
			},
			expectedHasCapacity: false,
		},
		{
			name: "GPUAvailable",
			setup: func(dockerManager *DockerManager, mockDockerClient *MockDockerClient) {
				// Mock client methods to simulate the image being available locally.
				mockDockerClient.On("ImageInspectWithRaw", mock.Anything, "default-image").Return(types.ImageInspect{}, []byte{}, nil)
				// Ensure that the GPU is available by not setting any container for the GPU.
				dockerManager.gpuContainers = make(map[string]string)
			},
			expectedHasCapacity: true,
		},
		{
			name: "GPUNotAvailable",
			setup: func(dockerManager *DockerManager, mockDockerClient *MockDockerClient) {
				// Mock client methods to simulate the image being available locally.
				mockDockerClient.On("ImageInspectWithRaw", mock.Anything, "default-image").Return(types.ImageInspect{}, []byte{}, nil)
				// Ensure that the GPU is not available by setting a container for the GPU.
				dockerManager.gpuContainers["gpu0"] = "container1"
			},
			expectedHasCapacity: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockDockerClient := new(MockDockerClient)
			dockerManager := createDockerManager(mockDockerClient)

			tt.setup(dockerManager, mockDockerClient)

			hasCapacity := dockerManager.HasCapacity(ctx, pipeline, modelID)
			require.Equal(t, tt.expectedHasCapacity, hasCapacity)

			mockDockerClient.AssertExpectations(t)
		})
	}
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
			mockDockerClient := new(MockDockerClient)
			manager := createDockerManager(mockDockerClient)

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
