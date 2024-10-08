package main

import (
	"flag"
	"io"
	"log"
)

type SegmentPoster struct {
	tw *TricklePublisher
}

func (sp *SegmentPoster) NewSegment(reader io.Reader) {
	go func() {
		// NB: This blocks! Very bad!
		sp.tw.Write(reader)
	}()
}

func main() {
	// Handle CLI options
	baseURL := flag.String("url", "http://localhost:2939", "Base URL for the trickle server")
	streamName := flag.String("stream", "", "Stream name (required)")
	localFile := flag.String("local", "", "Local file name (required)")
	flag.Parse()
	if *streamName == "" || *localFile == "" {
		log.Fatalf("Error: stream name and local file is required. Use -stream and `-local flags")
	}

	tw, err := NewTricklePublisher(*baseURL, *streamName)
	if err != nil {
		panic(err)
	}
	defer tw.Close()
	sp := &SegmentPoster{tw: tw}

	run(*localFile, sp)
}
