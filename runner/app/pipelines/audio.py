import os
import uuid
import av
from fastapi import UploadFile

class AudioConverter:
    def __init__(self):
        pass

    def m4a_to_mp3(self, upload_file: UploadFile) -> bytes:
        tempfile_out = self.generate_random_filename()

        input_container = av.open(upload_file.file)
        output_container = av.open(tempfile_out, mode='w')

        for stream in input_container.streams:
            if stream.type == 'audio':
                audio_stream = output_container.add_stream('mp3')

                # Loop through the frames in the input stream
                for frame in input_container.decode(stream):
                    # Encode the frame and write it to the output container
                    for packet in audio_stream.encode(frame):
                        output_container.mux(packet)

        # Finalize the encoding
        for packet in audio_stream.encode():
            output_container.mux(packet)

        input_container.close()
        output_container.close()

        # Read the converted file as bytes and delete when finished
        with open(tempfile_out, 'rb') as f:
            converted_bytes = f.read()
        os.remove(tempfile_out)

        return converted_bytes

    def generate_random_filename(self) -> str:
        filename = 'output_' + str(uuid.uuid4()) + '.mp3'
        while os.path.exists(filename):
            filename = 'output_' + str(uuid.uuid4()) + '.mp3'
        return filename

def __init__():
    pass
