package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"os"
)

// Subscribes to a given stream and outputs to stdout

func main() {

	// Check some command-line arguments
	baseURL := flag.String("url", "http://localhost:2939", "Base URL for the stream")
	streamName := flag.String("stream", "", "Stream name (required)")
	flag.Parse()
	if *streamName == "" {
		log.Fatalf("Error: stream name is required. Use -stream flag to specify the stream name.")
	}

	client := NewTrickleSubscriber(*baseURL, *streamName)

	maxSegments := 75
	for i := 0; i < maxSegments; i++ {
		// Read and process the first segment
		resp, err := client.Read()
		idx := getIndex(resp)
		if err != nil {
			log.Fatal("Failed to read segment", idx, err)
			continue
		}
		n, err := io.Copy(os.Stdout, resp.Body)
		if err != nil {
			log.Fatal("Failed to record segment", idx, err)
			continue
		}
		resp.Body.Close()
		log.Println("--- End of Segment ", idx, fmt.Sprintf("(%d/%d)", i, maxSegments), "bytes", humanBytes(n), " ---")
	}
	log.Println("Completing", *streamName)
}
