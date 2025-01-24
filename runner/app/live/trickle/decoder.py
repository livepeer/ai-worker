import sys
import av
from PIL import Image

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

    # We'll store decoded audio frames in a buffer if both audio and video exist.
    # If there's no video, we have the option to either:
    #   (A) call the callback for each audio frame, or
    #   (B) accumulate them and do something else.
    # Here, we'll do (A) for the audio-only case, and the original logic if video also exists.
    audio_buffer = []

    # Helper function to create a "result entry" for calling the callback
    def create_result_entry(
        video_pts=None,
        video_time=None,
        pil_img=None,
        matched_audio_frames=None,
        matched_audio_pts=None
    ):
        return {
            "video_pts": video_pts,
            "video_time_sec": video_time,
            "image": pil_img,  # None if no video
            "audio_frames": matched_audio_frames if matched_audio_frames else [],
            "audio_pts_list": matched_audio_pts if matched_audio_pts else [],
            "metadata": {
                # If we have a video frame, store width, height, etc.
                "width": pil_img.width if pil_img else None,
                "height": pil_img.height if pil_img else None,
                "pict_type": str(pil_img.info.get("pict_type")) if pil_img else None,
            },
            "audio_metadata": audio_metadata,
        }

    try:
        for packet in container.demux(streams_to_demux):
            if packet.dts is None:
                continue

            if audio_stream and packet.stream == audio_stream:
                # Decode audio frames
                for aframe in packet.decode():
                    if aframe.pts is None:
                        continue

                    if video_stream:
                        # If we also have video, buffer the audio frames
                        audio_buffer.append((aframe, aframe.pts))
                    else:
                        # If there's no video, we can call the callback immediately
                        # for each audio frame (audio-only use case).
                        # We set video_pts, image, etc. to None.
                        result_entry = create_result_entry(
                            video_pts=None,
                            video_time=None,
                            pil_img=None,
                            matched_audio_frames=[aframe],
                            matched_audio_pts=[aframe.pts],
                        )
                        frame_callback(result_entry)

            elif video_stream and packet.stream == video_stream:
                # Decode video frames
                for frame in packet.decode():
                    if frame.pts is None:
                        continue

                    video_pts = frame.pts
                    video_time = float(video_pts * video_stream.time_base)
                    pil_img = frame.to_image()

                    # If there's no audio stream, we can just call the callback with empty audio
                    if not audio_stream:
                        result_entry = create_result_entry(
                            video_pts=video_pts,
                            video_time=video_time,
                            pil_img=pil_img
                        )
                        frame_callback(result_entry)
                        continue

                    # Otherwise, gather audio frames up to this video_pts
                    matched_audio_frames = []
                    leftover_audio_buffer = []
                    for (aframe, apts) in audio_buffer:
                        if apts <= video_pts:
                            matched_audio_frames.append((aframe, apts))
                        else:
                            leftover_audio_buffer.append((aframe, apts))

                    # Remove matched frames from the buffer
                    audio_buffer = leftover_audio_buffer

                    # Build the callback entry
                    result_entry = create_result_entry(
                        video_pts=video_pts,
                        video_time=video_time,
                        pil_img=pil_img,
                        matched_audio_frames=[af[0] for af in matched_audio_frames],
                        matched_audio_pts=[af[1] for af in matched_audio_frames],
                    )
                    frame_callback(result_entry)

        # Optionally handle leftover audio frames if both audio and video exist
        # and you need to associate leftover audio with the final video frame, etc.

    except KeyboardInterrupt:
        print("Received Ctrl-C: stopping decode gracefully...")

    finally:
        container.close()

# ------------------------------------------------------------------------------
# Example usage:
def example_callback(result_entry):
    # This callback is invoked once per video frame (when video exists),
    # or once per audio frame (when no video exists).
    v_pts = result_entry["video_pts"]
    v_time = result_entry["video_time_sec"]
    n_audio = len(result_entry["audio_frames"])

    if v_pts is not None:
        print(
            f"Video frame PTS={v_pts} "
            f"(time={v_time:.3f}s), matched {n_audio} audio frames."
        )
    else:
        print(f"Audio-only frame, matched {n_audio} audio frames (PTS unknown for video).")

if __name__ == "__main__":
    # Example: reading from stdin. E.g., `python script.py < inputfile`
    decode_av(sys.stdin.buffer, example_callback, container_format="mov")
