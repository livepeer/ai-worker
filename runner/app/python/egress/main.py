import sys
import gi
import zmq
import argparse
from threading import Thread
import os
import time
import logging

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

class VideoEgress:
    def __init__(self, rtmp_url, zmq_address=None, framerate=30, width=1280, height=720, use_test_src=False):
        Gst.init(None)
        
        self.pipeline = Gst.Pipeline.new("video-egress")
        
        # Create elements
        if use_test_src:
            self.src = Gst.ElementFactory.make("videotestsrc", "test-source")
            self.src.set_property("is-live", True)
        else:
            self.src = Gst.ElementFactory.make("appsrc", "app-source")
            self.src.set_property("format", Gst.Format.TIME)
            self.src.set_property("is-live", True)
            
            # Add jpegdec element for JPEG decoding
            self.jpegdec = Gst.ElementFactory.make("jpegdec", "jpeg-decoder")

        self.videoconvert = Gst.ElementFactory.make("videoconvert", "video-convert")
        self.x264enc = Gst.ElementFactory.make("x264enc", "x264-encoder")
        self.h264parse = Gst.ElementFactory.make("h264parse", "h264-parser")
        self.flvmux = Gst.ElementFactory.make("flvmux", "flv-muxer")
        self.rtmpsink = Gst.ElementFactory.make("rtmpsink", "rtmp-sink")
        
        # Set caps for source
        if use_test_src:
            src_caps = Gst.Caps.from_string(f"video/x-raw,format=I420,width={width},height={height},framerate={framerate}/1")
        else:
            src_caps = Gst.Caps.from_string(f"image/jpeg,width={width},height={height},framerate={framerate}/1")
        self.capsfilter_src = Gst.ElementFactory.make("capsfilter", "src-caps")
        self.capsfilter_src.set_property("caps", src_caps)
        
        # Configure x264enc
        self.x264enc.set_property("tune", "zerolatency")
        self.x264enc.set_property("speed-preset", "ultrafast")
        self.x264enc.set_property("bitrate", 2000)
        self.x264enc.set_property("key-int-max", framerate * 2)  # GOP of 2 seconds
        self.x264enc.set_property("bframes", 0)  # No B-frames
        
        # Set H.264 profile using capsfilter after x264enc
        h264_caps = Gst.Caps.from_string("video/x-h264,profile=baseline")
        self.capsfilter_h264 = Gst.ElementFactory.make("capsfilter", "h264-caps")
        self.capsfilter_h264.set_property("caps", h264_caps)
        
        self.flvmux.set_property("streamable", True)
        self.rtmpsink.set_property("location", rtmp_url)
        
        # Add elements to pipeline
        if use_test_src:
            elements = [self.src, self.capsfilter_src, self.videoconvert, 
                        self.x264enc, self.capsfilter_h264, self.h264parse, self.flvmux, self.rtmpsink]
        else:
            elements = [self.src, self.capsfilter_src, self.jpegdec, self.videoconvert, 
                        self.x264enc, self.capsfilter_h264, self.h264parse, self.flvmux, self.rtmpsink]
        for element in elements:
            self.pipeline.add(element)
        
        # Link elements
        if use_test_src:
            self.src.link(self.capsfilter_src)
            self.capsfilter_src.link(self.videoconvert)
        else:
            self.src.link(self.capsfilter_src)
            self.capsfilter_src.link(self.jpegdec)
            self.jpegdec.link(self.videoconvert)
        self.videoconvert.link(self.x264enc)
        self.x264enc.link(self.capsfilter_h264)
        self.capsfilter_h264.link(self.h264parse)
        self.h264parse.link(self.flvmux)
        self.flvmux.link(self.rtmpsink)
        
        # Add debug probe
        self.add_probe(self.videoconvert.get_static_pad("src"), "after-videoconvert")

        if not use_test_src:
            # Set up ZMQ SUB socket
            self.context = zmq.Context()
            self.subscriber = self.context.socket(zmq.SUB)
            self.subscriber.bind(zmq_address)
            self.subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
            
            # Start ZMQ receiving thread
            self.running = True
            self.zmq_thread = Thread(target=self.receive_frames)
            self.zmq_thread.start()

        # Variables for input FPS calculation
        self.input_frame_count = 0
        self.input_start_time = time.time()

    def add_probe(self, pad, name):
        def probe_callback(pad, info, name):
            buffer = info.get_buffer()
            logging.debug(f"Probe {name}: Buffer pts={buffer.pts}, dts={buffer.dts}, duration={buffer.duration}, size={buffer.get_size()}")
            return Gst.PadProbeReturn.OK

        pad.add_probe(Gst.PadProbeType.BUFFER, probe_callback, name)

    def receive_frames(self):
        while self.running:
            try:
                frame = self.subscriber.recv(flags=zmq.NOBLOCK)
                buffer = Gst.Buffer.new_wrapped(frame)
                self.src.emit("push-buffer", buffer)
                
                # Update input frame count and calculate FPS
                self.input_frame_count += 1
                elapsed = time.time() - self.input_start_time
                if elapsed >= 5.0:  # Log every 5 seconds
                    input_fps = self.input_frame_count / elapsed
                    logging.info(f"Input FPS: {input_fps:.2f}")
                    self.input_frame_count = 0
                    self.input_start_time = time.time()
                
            except zmq.Again:
                continue

    def run(self):
        # Start playing
        self.pipeline.set_state(Gst.State.PLAYING)
        
        # Run the main loop
        loop = GLib.MainLoop()
        
        # Add a message handler to the pipeline bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_message, loop)
        
        try:
            loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            if hasattr(self, 'running'):
                self.running = False
                self.zmq_thread.join()
            self.pipeline.set_state(Gst.State.NULL)

    def on_message(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            logging.info("End-of-stream")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logging.error(f"Error: {err.message}")
            if debug:
                logging.debug(f"Debug info: {debug}")
            loop.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Egress to RTMP stream")
    parser.add_argument("--rtmp-host", type=str, help="RTMP host address", default="host.docker.internal")
    parser.add_argument("--rtmp-stream", type=str, help="RTMP stream name", required=True)
    parser.add_argument("--zmq-address", type=str, help="ZMQ address to subscribe to", default="tcp://localhost:5556")
    parser.add_argument("--framerate", type=int, help="Input framerate", default=30)
    parser.add_argument("--width", type=int, help="Video width", default=1280)
    parser.add_argument("--height", type=int, help="Video height", default=720)
    parser.add_argument("--use-test-src", action="store_true", help="Use videotestsrc instead of ZMQ input")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Construct RTMP URL
    rtmp_url = f"rtmp://{args.rtmp_host}/{args.rtmp_stream}"
    
    logging.info(f"Streaming to: {rtmp_url}")
    if not args.use_test_src:
        logging.info(f"Receiving from ZMQ: {args.zmq_address}")
    else:
        logging.info("Using videotestsrc")
    logging.info(f"Video settings: {args.width}x{args.height} at {args.framerate} fps")

    egress = VideoEgress(rtmp_url, args.zmq_address, args.framerate, args.width, args.height, args.use_test_src)
    egress.run()