package main

import (
	"fmt"
	"io"
	"log"
	"log/slog"
	"net/http"
	"os"
)

// Listens to new streams from MediaMTX and publishes
// to trickle HTTP server under the same name

type SegmentPoster struct {
	tricklePublisher *TricklePublisher
}

func (sp *SegmentPoster) NewSegment(reader io.Reader) {
	go func() {
		// NB: This blocks! Very bad!
		sp.tricklePublisher.Write(reader)
	}()
}

type FilePublisher struct {
	count int
}

func (fw *FilePublisher) NewSegment(reader io.Reader) {
	go func() {
		fname := fmt.Sprintf("out-write-rtsp/%d.ts", fw.count)
		file, err := os.OpenFile(fname, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
		if err != nil {
			panic(err)
		}
		defer file.Close()
		fw.count += 1
		io.Copy(file, reader)
	}()
}

func segmentPoster(streamName string) *SegmentPoster {
	c, err := NewTricklePublisher("http://localhost:2939", streamName)
	if err != nil {
		panic(err)
	}
	return &SegmentPoster{
		tricklePublisher: c,
	}
}

func listen(host string) {
	srv := &http.Server{
		Addr: host,
	}
	http.HandleFunc("POST /{streamName}/{$}", newPublish)
	log.Println("Listening for MediaMTX at ", host)
	log.Fatal(srv.ListenAndServe())
}

func newPublish(w http.ResponseWriter, r *http.Request) {
	streamName := r.PathValue("streamName")

	slog.Info("Starting stream", "streamName", streamName)

	go func() {
		//sp := segmentPoster(randomString())
		sp := segmentPoster(streamName)
		defer sp.tricklePublisher.Close()
		//sp := &FilePublisher{}
		//run("rtsp://localhost:8554/"+streamName+"?tcp", sp)
		run("rtmp://localhost/"+streamName, sp)
	}()
}

func main() {
	listen(":2938")
}
