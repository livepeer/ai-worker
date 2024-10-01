import sys
import gi
import zmq
import time
import argparse
import os
import errno
import logging
import signal
import threading
import io

# Disable output buffering
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, line_buffering=True)

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

class VideoIngress:
    def __init__(self, fd_or_path):
        Gst.init(None)
        
        self.pipeline = Gst.Pipeline.new("video-ingress")
        
        # Create elements
        self.src = Gst.ElementFactory.make("fdsrc", "fd-source")
        self.tsdemux = Gst.ElementFactory.make("tsdemux", "ts-demux")
        self.queue = Gst.ElementFactory.make("queue", "queue")
        self.decodebin = Gst.ElementFactory.make("decodebin", "decode-bin")
        self.videoconvert = Gst.ElementFactory.make("videoconvert", "video-convert")
        self.videoscale = Gst.ElementFactory.make("videoscale", "video-scale")
        self.capsfilter = Gst.ElementFactory.make("capsfilter", "caps-filter")
        self.jpegenc = Gst.ElementFactory.make("jpegenc", "jpeg-encoder")
        self.appsink = Gst.ElementFactory.make("appsink", "app-sink")
        
        # Set properties
        self.fd = self.open_fd(fd_or_path)
        self.src.set_property("fd", self.fd)
        self.queue.set_property("max-size-buffers", 1)
        self.queue.set_property("max-size-time", 0)
        self.queue.set_property("max-size-bytes", 0)
        self.capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw,width=512,height=512"))
        self.appsink.set_property("max-buffers", 1)
        self.appsink.set_property("drop", True)
        self.appsink.set_property("sync", False)
        
        # Add elements to pipeline
        elements = [self.src, self.tsdemux, self.queue, self.decodebin, 
                    self.videoconvert, self.videoscale, self.capsfilter, 
                    self.jpegenc, self.appsink]
        for element in elements:
            self.pipeline.add(element)
        
        # Link elements
        self.src.link(self.tsdemux)
        self.queue.link(self.decodebin)
        self.videoconvert.link(self.videoscale)
        self.videoscale.link(self.capsfilter)
        self.capsfilter.link(self.jpegenc)
        self.jpegenc.link(self.appsink)
        
        # Connect pad-added signals
        self.tsdemux.connect("pad-added", self.on_pad_added)
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

        self.fd_or_path = fd_or_path  # Store the original fd_or_path

    def open_fd(self, fd_or_path):
        if isinstance(fd_or_path, int):
            return fd_or_path
        else:
            while True:
                try:
                    return os.open(fd_or_path, os.O_RDONLY | os.O_NONBLOCK)
                except OSError as e:
                    if e.errno == errno.ENOENT:
                        print(f"Pipe {fd_or_path} not found. Waiting for it to be created...")
                        time.sleep(1)
                    else:
                        raise

    def on_pad_added(self, element, pad):
        if pad.get_direction() == Gst.PadDirection.SRC:
            if element == self.tsdemux:
                pad.link(self.queue.get_static_pad("sink"))
            elif element == self.decodebin:
                pad.link(self.videoconvert.get_static_pad("sink"))

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
        self.loop = GLib.MainLoop()
        
        # Add a message handler to the pipeline bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_message)
        
        try:
            self.loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.pipeline.set_state(Gst.State.NULL)
            os.close(self.fd)

    def on_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            logging.info("End-of-stream reached, restarting pipeline")
            self.restart_pipeline()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logging.error(f"Error: {err.message}")
            if debug:
                logging.debug(f"Debug info: {debug}")
            self.loop.quit()

    def restart_pipeline(self):
        logging.info("Restarting pipeline")
        # Stop the pipeline
        self.pipeline.set_state(Gst.State.NULL)
        
        # Close and reopen the file descriptor
        os.close(self.fd)
        self.fd = self.open_fd(self.fd_or_path)
        self.src.set_property("fd", self.fd)
        
        # Restart the pipeline
        self.pipeline.set_state(Gst.State.PLAYING)

def cleanup_with_timeout(ingress, timeout=5):
    def cleanup():
        if hasattr(ingress, 'pipeline'):
            ingress.pipeline.set_state(Gst.State.NULL)
        if hasattr(ingress, 'fd'):
            try:
                os.close(ingress.fd)
            except OSError:
                pass
        if hasattr(ingress, 'publisher'):
            ingress.publisher.close()
        if hasattr(ingress, 'context'):
            ingress.context.term()

    cleanup_thread = threading.Thread(target=cleanup)
    cleanup_thread.start()
    cleanup_thread.join(timeout)
    if cleanup_thread.is_alive():
        logging.warning("Cleanup timed out, forcing exit")

def main():
    parser = argparse.ArgumentParser(description="Video Ingress from MPEG-TS stream")
    parser.add_argument("--stream", help="File descriptor or path for the input stream", default="/tmp/video_pipe")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

    def signal_handler(sig, frame):
        logging.info("Received interrupt, shutting down...")
        nonlocal ingress
        if ingress:
            cleanup_with_timeout(ingress)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    while True:
        ingress = None
        try:
            logging.info(f"Initializing VideoIngress with stream: {args.stream}")
            ingress = VideoIngress(args.stream)
            ingress.run()
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            logging.info("Restarting in 5 seconds...")
            time.sleep(5)
        finally:
            if ingress:
                logging.info("Cleaning up resources...")
                cleanup_with_timeout(ingress)

if __name__ == "__main__":
    main()