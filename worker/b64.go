package worker

import (
	"bytes"
	"fmt"
	"image"
	"image/png"
	"os"

	"github.com/vincent-petithory/dataurl"
)

func SaveImageB64DataUrl(url, outputPath string) error {
	dataURL, err := dataurl.DecodeString(url)
	if err != nil {
		return err
	}

	img, _, err := image.Decode(bytes.NewReader(dataURL.Data))
	if err != nil {
		return err
	}

	file, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer file.Close()

	switch dataURL.MediaType.ContentType() {
	case "image/png":
		err = png.Encode(file, img)
		// Add cases for other image formats if necessary
	default:
		return fmt.Errorf("unsupported image format: %s", dataURL.MediaType.ContentType())
	}

	return err
}
