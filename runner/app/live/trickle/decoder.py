import sys
import av
from PIL import Image

from .frame import InputFrame

def decode_av(pipe_input, frame_callback, container_format=None):
    """
    Reads from a pipe (or file-like object). If both audio and video
    streams exist, for each decoded video frame, we gather all audio frames
    whose PTS is <= the video frame's PTS, then call `frame_callback`.

    Cases handled:
      - No audio (video only).
      - No video (audio only).
      - Both audio and video.

    :param pipe_input: File path, 'pipe:', sys.stdin, or another file-like object.
    :param frame_callback: A function that accepts a dictionary, e.g.:
        {
            'video_pts': int or None,
            'video_time_sec': float or None,
            'image': PIL.Image or None,
            'audio_frames': list of PyAV AudioFrame,
            'audio_pts_list': list of int,
            'metadata': {
                'width': int,
                'height': int,
                'pict_type': str,
                ...
            },
            'audio_metadata': dict or None  # e.g., sample_rate, channels, layout
        }
    :param container_format: Optional format hint for PyAV (e.g., 'mov', 'mp4', etc.).
    """
    container = av.open(pipe_input, format=container_format)

    # Locate the first video and first audio stream (if they exist)
    video_stream = None
    audio_stream = None
    for s in container.streams:
        if s.type == 'video' and video_stream is None:
            video_stream = s
        elif s.type == 'audio' and audio_stream is None:
            audio_stream = s

    # Prepare a list of streams to demux
    streams_to_demux = []
    if video_stream is not None:
        streams_to_demux.append(video_stream)
    if audio_stream is not None:
        streams_to_demux.append(audio_stream)

    if not streams_to_demux:
        print("No audio or video streams found in the input.")
        container.close()
        return

    # Prepare audio-related metadata (if audio is present)
    audio_metadata = None
    if audio_stream is not None:
        audio_metadata = {
            "codec": audio_stream.codec_context.name,
            "sample_rate": audio_stream.codec_context.sample_rate,
            "format": audio_stream.codec_context.format,
            "channels": audio_stream.codec_context.channels,
            "layout": str(audio_stream.layout),
            "time_base": str(audio_stream.time_base),
            "bit_rate": audio_stream.codec_context.bit_rate,
        }

    print(f"Audio metadata: {audio_metadata}")

    try:
        for packet in container.demux(streams_to_demux):
            if packet.dts is None:
                continue

            if audio_stream and packet.stream == audio_stream:
                # Decode audio frames
                for aframe in packet.decode():
                    if aframe.pts is None:
                        continue

                    avframe = InputFrame.from_av_audio(aframe)
                    frame_callback(avframe)
                    continue

            elif video_stream and packet.stream == video_stream:
                # Decode video frames
                for frame in packet.decode():
                    if frame.pts is None:
                        continue

                    avframe = InputFrame.from_av_video(frame)
                    frame_callback(avframe)
                    continue

    except KeyboardInterrupt:
        print("Received Ctrl-C: stopping decode gracefully...")

    finally:
        container.close()
