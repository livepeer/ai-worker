import io
import os
import sys
import logging

class JPEGStreamParser:
    def __init__(self, callback):
        """
        Initializes the JPEGStreamParser.
        :param callback: Function to be called when a complete JPEG is found. Receives bytes of the JPEG image.
        """
        self.buffer = bytearray()
        self.callback = callback
        self.in_jpeg = False
        self.start_idx = 0  # Keep track of the start index across feeds

    def __enter__(self):
        return self

    def __exit__(self, exec_type, exec_val, exec_tb):
        logging.info("JOSH closing jpeg parser via exit")
        self.close()

    def close(self):
        logging.info("Closing jpeg parser")
        self.callback(None)

    def feed(self, data):
        """
        Feed incoming data into the parser.
        :param data: Incoming data bytes.
        """
        self.buffer.extend(data)

        while True:
            # Find the start marker (0xFFD8) if not already in JPEG
            if not self.in_jpeg:
                self.start_idx = self.buffer.find(b'\xff\xd8', self.start_idx)
                if self.start_idx == -1:
                    # No start marker found, buffer the data for later.
                    self.start_idx = max(0, len(self.buffer) - 1)  # Move to the end of the buffer to prevent unnecessary re-scanning
                    return
                self.in_jpeg = True
                self.start_idx += 2  # Move past the start marker

            # Look for the end marker (0xFFD9)
            end_idx = self.buffer.find(b'\xff\xd9', self.start_idx)
            if end_idx == -1:
                # No end marker found, keep buffering
                self.start_idx = max(0, len(self.buffer) - 1)  # Move to the end of the buffer to prevent unnecessary re-scanning
                return

            # Extract the JPEG and call the callback
            jpeg_data = self.buffer[:end_idx + 2]
            self.callback(jpeg_data)

            # Remove the processed JPEG from the buffer
            self.buffer = self.buffer[end_idx + 2:]
            self.in_jpeg = False
            self.start_idx = 0  # Reset start_idx to begin from the start of the updated buffer
