package worker

import (
	"bytes"
	"encoding/base64"
	"os"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestReadImageB64DataUrl(t *testing.T) {
	// Create a sample PNG image and encode it as a data URL
	imgData := []byte{
		0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG header
		// ... (rest of the PNG data)
	}
	dataURL := "data:image/png;base64," + base64.StdEncoding.EncodeToString(imgData)

	var buf bytes.Buffer
	err := ReadImageB64DataUrl(dataURL, &buf)
	require.NoError(t, err)
	require.NotEmpty(t, buf.Bytes())
}

func TestSaveImageB64DataUrl(t *testing.T) {
	// Create a sample PNG image and encode it as a data URL
	imgData := []byte{
		0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG header
		// ... (rest of the PNG data)
	}
	dataURL := "data:image/png;base64," + base64.StdEncoding.EncodeToString(imgData)

	outputPath := "test_output.png"
	defer os.Remove(outputPath)

	err := SaveImageB64DataUrl(dataURL, outputPath)
	require.NoError(t, err)

	// Verify that the file was created and is not empty
	fileInfo, err := os.Stat(outputPath)
	require.NoError(t, err)
	require.False(t, fileInfo.IsDir())
	require.NotZero(t, fileInfo.Size())
}

func TestReadAudioB64DataUrl(t *testing.T) {
	// Create a sample audio data and encode it as a data URL
	audioData := []byte{0x00, 0x01, 0x02, 0x03, 0x04}
	dataURL := "data:audio/wav;base64," + base64.StdEncoding.EncodeToString(audioData)

	var buf bytes.Buffer
	err := ReadAudioB64DataUrl(dataURL, &buf)
	require.NoError(t, err)
	require.Equal(t, audioData, buf.Bytes())
}
