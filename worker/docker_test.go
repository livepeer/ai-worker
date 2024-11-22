package worker

import (
	"context"
	"errors"
	"io"
	"strings"
	"sync"
	"testing"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/image"
	"github.com/docker/docker/api/types/network"
	docker "github.com/docker/docker/client"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

// MockDockerClient is a mock implementation of the Docker client.
type MockDockerClient struct {
	docker.Client
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
		image, err := dockerManager.getContainerImage(tt.pipeline, tt.modelID)
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
	dockerManager := &DockerManager{
		dockerClient:    mockDockerClient,
		imagePullStatus: &sync.Map{},
	}

	ctx := context.Background()
	imageName := "test-image"

	// Mock the ImageInspectWithRaw method to simulate the image being available locally.
	mockDockerClient.On("ImageInspectWithRaw", ctx, imageName).Return(types.ImageInspect{}, nil, nil)

	// Call pullImageAsync and verify that it does not attempt to pull the image.
	dockerManager.pullImageAsync(ctx, imageName)
	mockDockerClient.AssertNotCalled(t, "ImagePull", ctx, imageName, mock.Anything)

	// Mock the ImageInspectWithRaw method to simulate the image not being available locally.
	mockDockerClient.On("ImageInspectWithRaw", ctx, imageName).Return(types.ImageInspect{}, nil, errors.New("not found"))

	// Mock the ImagePull method to simulate pulling the image.
	mockDockerClient.On("ImagePull", ctx, imageName, mock.Anything).Return(io.NopCloser(strings.NewReader("")), nil)

	// Call pullImageAsync and verify that it attempts to pull the image.
	dockerManager.pullImageAsync(ctx, imageName)
	mockDockerClient.AssertCalled(t, "ImagePull", ctx, imageName, mock.Anything)
}

func TestPullImagesAtStartup(t *testing.T) {
	mockDockerClient := new(MockDockerClient)
	dockerManager := &DockerManager{
		dockerClient:    mockDockerClient,
		imagePullStatus: &sync.Map{},
	}

	ctx := context.Background()

	// Mock the ImageInspectWithRaw method to simulate the images being available locally.
	for _, imageName := range pipelineToImage {
		mockDockerClient.On("ImageInspectWithRaw", ctx, imageName).Return(types.ImageInspect{}, nil, nil)
	}
	for _, imageName := range livePipelineToImage {
		mockDockerClient.On("ImageInspectWithRaw", ctx, imageName).Return(types.ImageInspect{}, nil, nil)
	}

	// Call pullImagesAtStartup and verify that it does not attempt to pull the images.
	dockerManager.pullImagesAtStartup(ctx)
	for _, imageName := range pipelineToImage {
		mockDockerClient.AssertNotCalled(t, "ImagePull", ctx, imageName, mock.Anything)
	}
	for _, imageName := range livePipelineToImage {
		mockDockerClient.AssertNotCalled(t, "ImagePull", ctx, imageName, mock.Anything)
	}
}

func TestPullImage(t *testing.T) {
	mockDockerClient := new(MockDockerClient)
	dockerManager := &DockerManager{
		dockerClient: mockDockerClient,
	}

	ctx := context.Background()
	imageName := "test-image"

	// Mock the ImagePull method to simulate pulling the image.
	mockDockerClient.On("ImagePull", ctx, imageName, mock.Anything).Return(io.NopCloser(strings.NewReader("")), nil)

	// Call pullImage and verify that it pulls the image.
	err := dockerManager.pullImage(ctx, imageName)
	require.NoError(t, err)
	mockDockerClient.AssertCalled(t, "ImagePull", ctx, imageName, mock.Anything)
}
