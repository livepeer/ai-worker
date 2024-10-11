import sys
import gi
import zmq
import argparse
from threading import Thread
import os

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

class VideoEgress:
    def __init__(self, rtmp_url, zmq_address):
        Gst.init(None)
        
        self.pipeline = Gst.Pipeline.new("video-egress")
        
        # Create elements
        self.appsrc = Gst.ElementFactory.make("appsrc", "app-source")
        self.jpegdec = Gst.ElementFactory.make("jpegdec", "jpeg-decoder")
        self.videoconvert = Gst.ElementFactory.make("videoconvert", "video-convert")
        self.x264enc = Gst.ElementFactory.make("x264enc", "x264-encoder")
        self.h264parse = Gst.ElementFactory.make("h264parse", "h264-parser")
        self.flvmux = Gst.ElementFactory.make("flvmux", "flv-muxer")
        self.rtmpsink = Gst.ElementFactory.make("rtmpsink", "rtmp-sink")
        
        # Set properties
        self.appsrc.set_property("format", Gst.Format.TIME)
        self.appsrc.set_property("is-live", True)
        self.x264enc.set_property("tune", "zerolatency")
        self.x264enc.set_property("speed-preset", "ultrafast")
        self.x264enc.set_property("bitrate", 2000)  # Adjust as needed
        self.flvmux.set_property("streamable", True)
        self.rtmpsink.set_property("location", rtmp_url)
        
        # Add elements to pipeline
        elements = [self.appsrc, self.jpegdec, self.videoconvert, 
                    self.x264enc, self.h264parse, self.flvmux, self.rtmpsink]
        for element in elements:
            self.pipeline.add(element)
        
        # Link elements
        self.appsrc.link(self.jpegdec)
        self.jpegdec.link(self.videoconvert)
        self.videoconvert.link(self.x264enc)
        self.x264enc.link(self.h264parse)
        self.h264parse.link(self.flvmux)
        self.flvmux.link(self.rtmpsink)
        
        # Set up ZMQ SUB socket
        self.context = zmq.Context()
        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.bind(zmq_address)
        self.subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        
        # Start ZMQ receiving thread
        self.running = True
        self.zmq_thread = Thread(target=self.receive_frames)
        self.zmq_thread.start()

    def receive_frames(self):
        while self.running:
            try:
                frame = self.subscriber.recv(flags=zmq.NOBLOCK)
                buffer = Gst.Buffer.new_wrapped(frame)
                self.appsrc.emit("push-buffer", buffer)
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
            self.running = False
            self.zmq_thread.join()
            self.pipeline.set_state(Gst.State.NULL)

    def on_message(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("End-of-stream")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err.message}")
            if debug:
                print(f"Debug info: {debug}")
            loop.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Egress to RTMP stream")
    parser.add_argument("--rtmp-host", type=str, help="RTMP host address", default="host.docker.internal")
    parser.add_argument("--rtmp-stream", type=str, help="RTMP stream name", required=True)
    parser.add_argument("--zmq-address", type=str, help="ZMQ address to subscribe to", default="tcp://localhost:5556")
    args = parser.parse_args()

    # Construct RTMP URL
    rtmp_url = f"rtmp://{args.rtmp_host}/{args.rtmp_stream}"
    
    print(f"Streaming to: {rtmp_url}")
    print(f"Receiving from ZMQ: {args.zmq_address}")

    egress = VideoEgress(rtmp_url, args.zmq_address)
    egress.run()
