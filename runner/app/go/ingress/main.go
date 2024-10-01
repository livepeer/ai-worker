package main

import (
	"fmt"
	"log"
	"time"

	"github.com/go-gst/go-gst/gst"
	"github.com/go-gst/go-gst/gst/app"
	"github.com/go-gst/go-glib/glib"
	zmq "github.com/pebbe/zmq4"
)

func main() {
	// Initialize GStreamer
	gst.Init(nil)

	// Create a new pipeline
	pipeline, err := gst.NewPipeline("")
	if err != nil {
		log.Fatalf("Failed to create pipeline: %s", err)
	}

	// Create elements
	src, err := gst.NewElement("filesrc")
	if err != nil {
		log.Fatalf("Failed to create filesrc: %s", err)
	}
	src.SetProperty("location", "10s.mp4")  // Replace with your MP4 file path

	decodebin, err := gst.NewElement("decodebin")
	if err != nil {
		log.Fatalf("Failed to create decodebin: %s", err)
	}

	queue, err := gst.NewElement("queue")
	if err != nil {
		log.Fatalf("Failed to create queue: %s", err)
	}
	queue.SetProperty("max-size-buffers", uint(1))
	queue.SetProperty("max-size-time", uint64(0))
	queue.SetProperty("max-size-bytes", uint(0))

	videoconvert, err := gst.NewElement("videoconvert")
	if err != nil {
		log.Fatalf("Failed to create videoconvert: %s", err)
	}

	videoscale, err := gst.NewElement("videoscale")
	if err != nil {
		log.Fatalf("Failed to create videoscale: %s", err)
	}

	capsfilter, err := gst.NewElement("capsfilter")
	if err != nil {
		log.Fatalf("Failed to create capsfilter: %s", err)
	}
	capsfilter.SetProperty("caps", gst.NewCapsFromString("video/x-raw,width=512,height=512"))

	jpegenc, err := gst.NewElement("jpegenc")
	if err != nil {
		log.Fatalf("Failed to create jpegenc: %s", err)
	}

	appsink, err := app.NewAppSink()
	if err != nil {
		log.Fatalf("Failed to create appsink: %s", err)
	}
	appsink.SetProperty("max-buffers", uint(1))
	appsink.SetProperty("drop", true)
	appsink.SetProperty("sync", false)

	// Add elements to pipeline
	pipeline.AddMany(src, decodebin, queue, videoconvert, videoscale, capsfilter, jpegenc, appsink.Element)

	// Link elements
	src.Link(decodebin)
	queue.Link(videoconvert)
	videoconvert.Link(videoscale)
	videoscale.Link(capsfilter)
	capsfilter.Link(jpegenc)
	jpegenc.Link(appsink.Element)

	// Connect pad-added signal for decodebin
	decodebin.Connect("pad-added", func(element *gst.Element, pad *gst.Pad) {
		sinkpad := queue.GetStaticPad("sink")
		pad.Link(sinkpad)
	})

	// Set up ZMQ PUB socket
	publisher, err := zmq.NewSocket(zmq.PUB)
	if err != nil {
		log.Fatalf("Failed to create ZMQ socket: %s", err)
	}
	defer publisher.Close()

	// Set high water mark
	err = publisher.SetSndhwm(1)
	if err != nil {
		log.Fatalf("Failed to set high water mark: %s", err)
	}

	err = publisher.Bind("tcp://*:5555")
	if err != nil {
		log.Fatalf("Failed to bind ZMQ socket: %s", err)
	}

	// Start playing
	pipeline.SetState(gst.StatePlaying)

	// Variables for frame rate calculation
	frameCount := 0
	startTime := time.Now()

	// Callback function for new samples
	appsink.SetCallbacks(&app.SinkCallbacks{
		NewSampleFunc: func(sink *app.Sink) gst.FlowReturn {
			sample := sink.PullSample()
			if sample == nil {
				log.Printf("Failed to pull sample")
				return gst.FlowError
			}

			buffer := sample.GetBuffer()
			data := buffer.Bytes()

			// Send frame via ZMQ
			_, err = publisher.SendBytes(data, zmq.DONTWAIT)
			if err != nil {
				log.Printf("Failed to send frame: %s", err)
			}

			// Update frame count and calculate FPS
			frameCount++
			elapsed := time.Since(startTime)
			if elapsed >= time.Second {
				fps := float64(frameCount) / elapsed.Seconds()
				log.Printf("Producer FPS: %.2f", fps)
				frameCount = 0
				startTime = time.Now()
			}

			return gst.FlowOK
		},
	})

	// Create a main loop
	mainLoop := glib.NewMainLoop(glib.MainContextDefault(), false)

	// Add a message handler to the pipeline bus
	pipeline.GetPipelineBus().AddWatch(func(msg *gst.Message) bool {
		switch msg.Type() {
		case gst.MessageEOS: // When end-of-stream is received flush the pipeline and stop the main loop
			pipeline.BlockSetState(gst.StateNull)
			mainLoop.Quit()
		case gst.MessageError: // Error messages are always fatal
			err := msg.ParseError()
			fmt.Println("ERROR:", err.Error())
			if debug := err.DebugString(); debug != "" {
				fmt.Println("DEBUG:", debug)
			}
			mainLoop.Quit()
		default:
			// All messages implement a Stringer. However, this is
			// typically an expensive thing to do and should be avoided.
			fmt.Println(msg)
		}
		return true
	})

	// Run the main loop
	mainLoop.Run()
}