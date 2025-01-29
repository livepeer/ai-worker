import asyncio
import av
import datetime
import logging
import os
from typing import Optional
from fractions import Fraction

from .frame import VideoOutput, AudioOutput

# microseconds in a second
US_IN_SECS=1_000_000
GOP_SECS=3

def encode_av(
    input_queue,
    output_callback,
    video_codec: Optional[str] ='libx264',
    audio_codec: Optional[str] ='libfdk_aac'
):
    logging.info("Starting encoder")
    av.logging.set_level(logging.DEBUG)

    def custom_io_open(url: str, flags: int, options: dict):
        read_fd, write_fd = os.pipe()
        read_file  = os.fdopen(read_fd,  'rb', buffering=0)
        write_file = os.fdopen(write_fd, 'wb', buffering=0)
        output_callback(read_file, url)
        return write_file

    # Open the output container in write mode
    output_container = av.open("%d.ts", format='segment', mode='w', io_open=custom_io_open)

    # Create corresponding output streams if input streams exist
    output_video_stream = None
    output_audio_stream = None

    if video_codec:
        # Add a new stream to the output using the desired video codec
        #video_opts = { 'width':512, 'height':512, 'bf':'0' }
        video_opts = { 'width':'512', 'height':'512', 'bf':'0' }
        if video_codec == 'libx264':
            video_opts = video_opts | { 'preset':'superfast', 'tune':'zerolatency' }
        output_video_stream = output_container.add_stream(video_codec, options=video_opts)
        output_video_stream.time_base = Fraction(1, US_IN_SECS)

        # Optional: set other encoding parameters, e.g.:
        # output_video_stream.bit_rate = 2_000_000  # 2 Mbps
        # output_video_stream.width = input_video_stream.codec_context.width
        # output_video_stream.height = input_video_stream.codec_context.height
        # output_video_stream.pix_fmt = 'yuv420p'  # example pix_fmt (depends on the codec)

    if audio_codec:
        # Add a new stream to the output using the desired audio codec
        output_audio_stream = output_container.add_stream(audio_codec)
        output_audio_stream.time_base = Fraction(1, US_IN_SECS)
        # Optional: set other encoding parameters, e.g.:
        # output_audio_stream.bit_rate = 128_000
        # output_audio_stream.sample_rate = input_audio_stream.codec_context.sample_rate
        # output_audio_stream.channels = input_audio_stream.codec_context.channels
        # output_audio_stream.layout = input_audio_stream.layout

    # Now read packets from the input, decode, then re-encode, and mux.
    start = datetime.datetime.now()
    last_kf = None
    while True:
        avframe = input_queue()
        if avframe is None:
            break

        if isinstance(avframe, VideoOutput):
            if not output_video_stream:
                # received video but no video output, so drop
                continue
            frame = av.video.frame.VideoFrame.from_image(avframe.image)
            frame.pts = rescale_ts(avframe.timestamp, Fraction(avframe.time_base), output_video_stream.codec_context.time_base)
            frame.time_base = output_video_stream.codec_context.time_base
            current = avframe.timestamp * avframe.time_base
            if not last_kf or (current - last_kf) > GOP_SECS:
                frame.pict_type = av.video.frame.PictureType.I
                last_kf = current
            encoded_packets = output_video_stream.encode(frame)
            for ep in encoded_packets:
                output_container.mux(ep)
            continue

        if isinstance(avframe, AudioOutput):
            if not output_audio_stream:
                # received audio but no audio output, so drop
                continue
            for af in avframe.frames:
                frame = av.audio.frame.AudioFrame.from_ndarray(af.samples, format=af.format, layout=af.layout)
                frame.sample_rate = af.rate
                frame.pts = rescale_ts(af.timestamp, Fraction(af.time_base), output_audio_stream.codec_context.time_base)
                frame.time_base = output_audio_stream.codec_context.time_base
                encoded_packets = output_audio_stream.encode(frame)
                for ep in encoded_packets:
                    output_container.mux(ep)
            continue

        logging.warning(f"Unsupported output frame type {type(avframe)}")

    # After reading all packets, flush encoders
    logging.info("Stopping encoder")
    if output_video_stream:
        encoded_packets = output_video_stream.encode(None)
        for ep in encoded_packets:
            output_container.mux(ep)

    if output_audio_stream:
        encoded_packets = output_audio_stream.encode(None)
        for ep in encoded_packets:
            output_container.mux(ep)

    # Close the output container to finish writing
    output_container.close()

def rescale_ts(pts: int, orig_tb: Fraction, dest_tb: Fraction):
    return int(round(float((Fraction(pts) * orig_tb) / dest_tb)))
