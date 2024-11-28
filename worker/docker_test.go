package worker

import (
	"context"
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

// createDockerManager creates a DockerManager with a mock DockerClient.
func createDockerManager(mockDockerClient *MockDockerClient) *DockerManager {
	return &DockerManager{
		defaultImage:  "default-image",
		gpus:          []string{"gpu0"},
		modelDir:      "/models",
		dockerClient:  mockDockerClient,
		gpuContainers: make(map[string]string),
		containers:    make(map[string]*RunnerContainer),
		mu:            &sync.Mutex{},
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

func TestDockerManager_EnsureImageAvailable(t *testing.T) {
	mockDockerClient := new(MockDockerClient)
	dockerManager := createDockerManager(mockDockerClient)

	ctx := context.Background()
	pipeline := "text-to-image"
	modelID := "test-model"

	tests := []struct {
		name         string
		setup        func(*DockerManager, *MockDockerClient)
		expectedPull bool
	}{
		{
			name: "ImageAvailable",
			setup: func(dockerManager *DockerManager, mockDockerClient *MockDockerClient) {
				// Mock client methods to simulate the image being available locally.
				mockDockerClient.On("ImageInspectWithRaw", mock.Anything, "default-image").Return(types.ImageInspect{}, []byte{}, nil).Once()
			},
			expectedPull: false,
		},
		{
			name: "ImageNotAvailable",
			setup: func(dockerManager *DockerManager, mockDockerClient *MockDockerClient) {
				// Mock client methods to simulate the image not being available locally.
				mockDockerClient.On("ImageInspectWithRaw", mock.Anything, "default-image").Return(types.ImageInspect{}, []byte{}, fmt.Errorf("image not found")).Once()
			},
			expectedPull: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.setup(dockerManager, mockDockerClient)

			if tt.expectedPull {
				mockDockerClient.On("ImagePull", mock.Anything, "default-image", mock.Anything).Return(io.NopCloser(strings.NewReader("")), nil).Once()
			}

			err := dockerManager.EnsureImageAvailable(ctx, pipeline, modelID)
			require.NoError(t, err)

			mockDockerClient.AssertExpectations(t)
		})
	}
}

func TestDockerManager_Warm(t *testing.T) {
	mockDockerClient := new(MockDockerClient)
	dockerManager := createDockerManager(mockDockerClient)

	ctx := context.Background()
	pipeline := "text-to-image"
	modelID := "test-model"
	containerID := "container1"
	optimizationFlags := OptimizationFlags{}

	// Mock nested functions.
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

	mockDockerClient.On("ContainerCreate", mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(container.CreateResponse{ID: containerID}, nil)
	mockDockerClient.On("ContainerStart", mock.Anything, containerID, mock.Anything).Return(nil)
	err := dockerManager.Warm(ctx, pipeline, modelID, optimizationFlags)
	require.NoError(t, err)
	mockDockerClient.AssertExpectations(t)
}

func TestDockerManager_Stop(t *testing.T) {
	MockDockerClient := new(MockDockerClient)
	dockerManager := createDockerManager(MockDockerClient)

	ctx, cancel := context.WithTimeout(context.Background(), containerRemoveTimeout)
	defer cancel()
	containerID := "container1"
	dockerManager.containers[containerID] = &RunnerContainer{
		RunnerContainerConfig: RunnerContainerConfig{
			ID: containerID,
		},
	}

	MockDockerClient.On("ContainerStop", mock.Anything, containerID, container.StopOptions{Timeout: nil}).Return(nil)
	MockDockerClient.On("ContainerRemove", mock.Anything, containerID, container.RemoveOptions{}).Return(nil)
	err := dockerManager.Stop(ctx)
	require.NoError(t, err)
	MockDockerClient.AssertExpectations(t)
}

func TestDockerManager_Borrow(t *testing.T) {
	mockDockerClient := new(MockDockerClient)
	dockerManager := createDockerManager(mockDockerClient)

	ctx := context.Background()
	pipeline := "text-to-image"
	modelID := "model"
	containerID, _ := dockerManager.getContainerImageName(pipeline, modelID)

	// Mock nested functions.
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

	mockDockerClient.On("ContainerCreate", mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(container.CreateResponse{ID: containerID}, nil)
	mockDockerClient.On("ContainerStart", mock.Anything, containerID, mock.Anything).Return(nil)
	rc, err := dockerManager.Borrow(ctx, pipeline, modelID)
	require.NoError(t, err)
	require.NotNil(t, rc)
	require.Empty(t, dockerManager.containers, "containers map should be empty")
	mockDockerClient.AssertExpectations(t)
}

func TestDockerManager_returnContainer(t *testing.T) {
	mockDockerClient := new(MockDockerClient)
	dockerManager := createDockerManager(mockDockerClient)

	// Create a RunnerContainer to return to the pool
	rc := &RunnerContainer{
		Name:                  "container1",
		RunnerContainerConfig: RunnerContainerConfig{},
	}

	// Ensure the container is not in the pool initially.
	_, exists := dockerManager.containers[rc.Name]
	require.False(t, exists)

	// Return the container to the pool.
	dockerManager.returnContainer(rc)

	// Verify the container is now in the pool.
	returnedContainer, exists := dockerManager.containers[rc.Name]
	require.True(t, exists)
	require.Equal(t, rc, returnedContainer)
}

func TestDockerManager_getContainerImageName(t *testing.T) {
	mockDockerClient := new(MockDockerClient)
	manager := createDockerManager(mockDockerClient)

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
				// Add an unused managed container.
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
				mockDockerClient.On("ImageInspectWithRaw", mock.Anything, "default-image").Return(types.ImageInspect{}, []byte{}, fmt.Errorf("image not found"))
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
			name: "GPUUnavailable",
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

func TestDockerManager_isImageAvailable(t *testing.T) {
	mockDockerClient := new(MockDockerClient)
	dockerManager := createDockerManager(mockDockerClient)

	ctx := context.Background()
	pipeline := "text-to-image"
	modelID := "test-model"

	t.Run("ImageNotFound", func(t *testing.T) {
		mockDockerClient.On("ImageInspectWithRaw", mock.Anything, "default-image").Return(types.ImageInspect{}, []byte{}, fmt.Errorf("image not found")).Once()

		isAvailable := dockerManager.isImageAvailable(ctx, pipeline, modelID)
		require.False(t, isAvailable)
		mockDockerClient.AssertExpectations(t)
	})

	t.Run("ImageFound", func(t *testing.T) {
		mockDockerClient.On("ImageInspectWithRaw", mock.Anything, "default-image").Return(types.ImageInspect{}, []byte{}, nil).Once()

		isAvailable := dockerManager.isImageAvailable(ctx, pipeline, modelID)
		require.True(t, isAvailable)
		mockDockerClient.AssertExpectations(t)
	})
}

func TestDockerManager_pullImage(t *testing.T) {
	mockDockerClient := new(MockDockerClient)
	dockerManager := createDockerManager(mockDockerClient)

	ctx := context.Background()
	imageName := "default-image"

	t.Run("ImagePullError", func(t *testing.T) {
		mockDockerClient.On("ImagePull", mock.Anything, imageName, mock.Anything).Return(io.NopCloser(strings.NewReader("")), fmt.Errorf("failed to pull image: pull error")).Once()

		err := dockerManager.pullImage(ctx, imageName)
		require.Error(t, err)
		require.Contains(t, err.Error(), "failed to pull image: pull error")
		mockDockerClient.AssertExpectations(t)
	})

	t.Run("ImagePullSuccess", func(t *testing.T) {
		mockDockerClient.On("ImagePull", mock.Anything, imageName, mock.Anything).Return(io.NopCloser(strings.NewReader("")), nil).Once()

		err := dockerManager.pullImage(ctx, imageName)
		require.NoError(t, err)
		mockDockerClient.AssertExpectations(t)
	})
}

func TestDockerManager_createContainer(t *testing.T) {
	mockDockerClient := new(MockDockerClient)
	dockerManager := createDockerManager(mockDockerClient)

	ctx := context.Background()
	pipeline := "text-to-image"
	modelID := "test-model"
	containerID := "container1"
	gpu := "0"
	containerHostPort := "8000"
	containerName := dockerContainerName(pipeline, modelID, containerHostPort)
	containerImage := "default-image"
	optimizationFlags := OptimizationFlags{}

	// Mock nested functions.
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

	// Mock allocGPU and getContainerImageName methods.
	dockerManager.gpus = []string{gpu}
	dockerManager.gpuContainers = make(map[string]string)
	dockerManager.containers = make(map[string]*RunnerContainer)
	dockerManager.defaultImage = containerImage

	mockDockerClient.On("ContainerCreate", mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(container.CreateResponse{ID: containerID}, nil)
	mockDockerClient.On("ContainerStart", mock.Anything, containerID, mock.Anything).Return(nil)

	rc, err := dockerManager.createContainer(ctx, pipeline, modelID, false, optimizationFlags)
	require.NoError(t, err)
	require.NotNil(t, rc)
	require.Equal(t, containerID, rc.ID)
	require.Equal(t, gpu, rc.GPU)
	require.Equal(t, pipeline, rc.Pipeline)
	require.Equal(t, modelID, rc.ModelID)
	require.Equal(t, containerName, rc.Name)

	mockDockerClient.AssertExpectations(t)
}

func TestDockerManager_allocGPU(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name                 string
		setup                func(*DockerManager, *MockDockerClient)
		expectedAllocatedGPU string
		errorMessage         string
	}{
		{
			name: "GPUAvailable",
			setup: func(dockerManager *DockerManager, mockDockerClient *MockDockerClient) {
				// Ensure that the GPU is available by not setting any container for the GPU.
				dockerManager.gpuContainers = make(map[string]string)
			},
			expectedAllocatedGPU: "gpu0",
			errorMessage:         "",
		},
		{
			name: "GPUUnavailable",
			setup: func(dockerManager *DockerManager, mockDockerClient *MockDockerClient) {
				// Ensure that the GPU is not available by setting a container for the GPU.
				dockerManager.gpuContainers["gpu0"] = "container1"
			},
			expectedAllocatedGPU: "",
			errorMessage:         "insufficient capacity",
		},
		{
			name: "GPUUnavailableAndWarm",
			setup: func(dockerManager *DockerManager, mockDockerClient *MockDockerClient) {
				// Ensure that the GPU is not available by setting a container for the GPU.
				dockerManager.gpuContainers["gpu0"] = "container1"
				dockerManager.containers["container1"] = &RunnerContainer{
					RunnerContainerConfig: RunnerContainerConfig{
						ID:       "container1",
						KeepWarm: true,
					},
				}
			},
			expectedAllocatedGPU: "",
			errorMessage:         "insufficient capacity",
		},
		{
			name: "GPUUnavailableButCold",
			setup: func(dockerManager *DockerManager, mockDockerClient *MockDockerClient) {
				// Ensure that the GPU is not available by setting a container for the GPU.
				dockerManager.gpuContainers["gpu0"] = "container1"
				dockerManager.containers["container1"] = &RunnerContainer{
					RunnerContainerConfig: RunnerContainerConfig{
						ID:       "container1",
						KeepWarm: false,
					},
				}
				// Mock client methods to simulate the removal of the warm container.
				mockDockerClient.On("ContainerStop", mock.Anything, "container1", container.StopOptions{}).Return(nil)
				mockDockerClient.On("ContainerRemove", mock.Anything, "container1", container.RemoveOptions{}).Return(nil)
			},
			expectedAllocatedGPU: "gpu0",
			errorMessage:         "",
		},
	}

	for _, tt := range tests {
		mockDockerClient := new(MockDockerClient)
		dockerManager := createDockerManager(mockDockerClient)

		tt.setup(dockerManager, mockDockerClient)

		gpu, err := dockerManager.allocGPU(ctx)
		if tt.errorMessage != "" {
			require.Error(t, err)
			require.Contains(t, err.Error(), tt.errorMessage)
		} else {
			require.NoError(t, err)
			require.Equal(t, tt.expectedAllocatedGPU, gpu)
		}
	}
}

func TestDockerManager_destroyContainer(t *testing.T) {
	mockDockerClient := new(MockDockerClient)
	dockerManager := createDockerManager(mockDockerClient)

	containerID := "container1"
	gpu := "gpu0"

	rc := &RunnerContainer{
		Name: containerID,
		RunnerContainerConfig: RunnerContainerConfig{
			ID:  containerID,
			GPU: gpu,
		},
	}
	dockerManager.gpuContainers[gpu] = containerID
	dockerManager.containers[containerID] = rc

	mockDockerClient.On("ContainerStop", mock.Anything, containerID, container.StopOptions{}).Return(nil)
	mockDockerClient.On("ContainerRemove", mock.Anything, containerID, container.RemoveOptions{}).Return(nil)

	err := dockerManager.destroyContainer(rc, true)
	require.NoError(t, err)
	require.Empty(t, dockerManager.gpuContainers, "gpuContainers map should be empty")
	require.Empty(t, dockerManager.containers, "containers map should be empty")
	mockDockerClient.AssertExpectations(t)
}

func TestDockerManager_watchContainer(t *testing.T) {
	mockDockerClient := new(MockDockerClient)
	dockerManager := createDockerManager(mockDockerClient)

	// Override the containerWatchInterval for testing purposes.
	containerWatchInterval = 10 * time.Millisecond

	containerID := "container1"
	rc := &RunnerContainer{
		Name: containerID,
		RunnerContainerConfig: RunnerContainerConfig{
			ID: containerID,
		},
	}

	t.Run("ReturnContainerOnContextDone", func(t *testing.T) {
		borrowCtx, cancel := context.WithCancel(context.Background())

		go dockerManager.watchContainer(rc, borrowCtx)
		cancel()                          // Cancel the context.
		time.Sleep(50 * time.Millisecond) // Ensure the ticker triggers.

		// Verify that the container was returned.
		_, exists := dockerManager.containers[rc.Name]
		require.True(t, exists)
	})

	t.Run("DestroyContainerOnNotRunning", func(t *testing.T) {
		borrowCtx := context.Background()

		// Mock ContainerInspect to return a non-running state.
		mockDockerClient.On("ContainerInspect", mock.Anything, containerID).Return(types.ContainerJSON{
			ContainerJSONBase: &types.ContainerJSONBase{
				State: &types.ContainerState{
					Running: false,
				},
			},
		}, nil).Once()

		// Mock destroyContainer to verify it is called.
		mockDockerClient.On("ContainerStop", mock.Anything, containerID, mock.Anything).Return(nil)
		mockDockerClient.On("ContainerRemove", mock.Anything, containerID, mock.Anything).Return(nil)

		go dockerManager.watchContainer(rc, borrowCtx)
		time.Sleep(50 * time.Millisecond) // Ensure the ticker triggers.

		// Verify that the container was destroyed.
		_, exists := dockerManager.containers[rc.Name]
		require.False(t, exists)
	})
}

// Watch container

func TestRemoveExistingContainers(t *testing.T) {
	mockDockerClient := new(MockDockerClient)

	ctx := context.Background()

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

	removeExistingContainers(ctx, mockDockerClient)
	mockDockerClient.AssertExpectations(t)
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

func TestDockerRemoveContainer(t *testing.T) {
	mockDockerClient := new(MockDockerClient)

	mockDockerClient.On("ContainerStop", mock.Anything, "container1", container.StopOptions{}).Return(nil)
	mockDockerClient.On("ContainerRemove", mock.Anything, "container1", container.RemoveOptions{}).Return(nil)

	err := dockerRemoveContainer(mockDockerClient, "container1")
	require.NoError(t, err)
	mockDockerClient.AssertExpectations(t)
}

func TestDockerWaitUntilRunning(t *testing.T) {
	mockDockerClient := new(MockDockerClient)
	containerID := "container1"
	pollingInterval := 10 * time.Millisecond
	ctx := context.Background()

	t.Run("ContainerRunning", func(t *testing.T) {
		// Mock ContainerInspect to return a running container state.
		mockDockerClient.On("ContainerInspect", mock.Anything, containerID).Return(types.ContainerJSON{
			ContainerJSONBase: &types.ContainerJSONBase{
				State: &types.ContainerState{
					Running: true,
				},
			},
		}, nil).Once()

		err := dockerWaitUntilRunning(ctx, mockDockerClient, containerID, pollingInterval)
		require.NoError(t, err)
		mockDockerClient.AssertExpectations(t)
	})

	t.Run("ContainerNotRunningInitially", func(t *testing.T) {
		// Mock ContainerInspect to return a non-running state initially, then a running state.
		mockDockerClient.On("ContainerInspect", mock.Anything, containerID).Return(types.ContainerJSON{
			ContainerJSONBase: &types.ContainerJSONBase{
				State: &types.ContainerState{
					Running: false,
				},
			},
		}, nil).Once()
		mockDockerClient.On("ContainerInspect", mock.Anything, containerID).Return(types.ContainerJSON{
			ContainerJSONBase: &types.ContainerJSONBase{
				State: &types.ContainerState{
					Running: true,
				},
			},
		}, nil).Once()

		err := dockerWaitUntilRunning(ctx, mockDockerClient, containerID, pollingInterval)
		require.NoError(t, err)
		mockDockerClient.AssertExpectations(t)
	})

	t.Run("ContextTimeout", func(t *testing.T) {
		// Create a context that will timeout.
		timeoutCtx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
		defer cancel()

		// Mock ContainerInspect to always return a non-running state.
		mockDockerClient.On("ContainerInspect", mock.Anything, containerID).Return(types.ContainerJSON{
			ContainerJSONBase: &types.ContainerJSONBase{
				State: &types.ContainerState{
					Running: false,
				},
			},
		}, nil)

		err := dockerWaitUntilRunning(timeoutCtx, mockDockerClient, containerID, pollingInterval)
		require.Error(t, err)
		require.Contains(t, err.Error(), "timed out waiting for managed container")
		mockDockerClient.AssertExpectations(t)
	})
}
