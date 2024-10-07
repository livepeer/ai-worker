import sys
import gi
import zmq
import time

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

class VideoIngress:
    def __init__(self):
        Gst.init(None)
        
        self.pipeline = Gst.Pipeline.new("video-ingress")
        
        # Create elements
        self.src = Gst.ElementFactory.make("filesrc", "file-source")
        self.decodebin = Gst.ElementFactory.make("decodebin", "decode-bin")
        self.queue = Gst.ElementFactory.make("queue", "queue")
        self.videoconvert = Gst.ElementFactory.make("videoconvert", "video-convert")
        self.videoscale = Gst.ElementFactory.make("videoscale", "video-scale")
        self.capsfilter = Gst.ElementFactory.make("capsfilter", "caps-filter")
        self.jpegenc = Gst.ElementFactory.make("jpegenc", "jpeg-encoder")
        self.appsink = Gst.ElementFactory.make("appsink", "app-sink")
        
        # Set properties
        self.src.set_property("location", "10s.mp4")  # Replace with your MP4 file path
        self.queue.set_property("max-size-buffers", 1)
        self.queue.set_property("max-size-time", 0)
        self.queue.set_property("max-size-bytes", 0)
        self.capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw,width=512,height=512"))
        self.appsink.set_property("max-buffers", 1)
        self.appsink.set_property("drop", True)
        self.appsink.set_property("sync", False)
        
        # Add elements to pipeline
        self.pipeline.add(self.src)
        self.pipeline.add(self.decodebin)
        self.pipeline.add(self.queue)
        self.pipeline.add(self.videoconvert)
        self.pipeline.add(self.videoscale)
        self.pipeline.add(self.capsfilter)
        self.pipeline.add(self.jpegenc)
        self.pipeline.add(self.appsink)
        
        # Link elements
        self.src.link(self.decodebin)
        self.queue.link(self.videoconvert)
        self.videoconvert.link(self.videoscale)
        self.videoscale.link(self.capsfilter)
        self.capsfilter.link(self.jpegenc)
        self.jpegenc.link(self.appsink)
        
        # Connect pad-added signal for decodebin
        self.decodebin.connect("pad-added", self.on_pad_added)
        
        # Set up ZMQ PUB socket
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.set_hwm(1)
        self.publisher.bind("tcp://*:5555")
        
        # Set up appsink callbacks
        self.appsink.set_property('emit-signals', True)
        self.appsink.connect("new-sample", self.on_new_sample)
        
        # Variables for frame rate calculation
        self.frame_count = 0
        self.start_time = time.time()

    def on_pad_added(self, element, pad):
        sink_pad = self.queue.get_static_pad("sink")
        pad.link(sink_pad)

    def on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample:
            buffer = sample.get_buffer()
            data = buffer.extract_dup(0, buffer.get_size())
            
            # Send frame via ZMQ
            try:
                self.publisher.send(data, zmq.NOBLOCK)
            except zmq.error.Again:
                print("Failed to send frame: High water mark reached")
            
            # Update frame count and calculate FPS
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            if elapsed >= 1.0:
                fps = self.frame_count / elapsed
                print(f"Producer FPS: {fps:.2f}")
                self.frame_count = 0
                self.start_time = time.time()
        
        return Gst.FlowReturn.OK

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
    ingress = VideoIngress()
    ingress.run()