import glob
import os
from PIL import Image
import torch
from torchvision.transforms import v2
from torchaudio.io import StreamWriter
from typing import List


class ListReader:
    def __init__(self, frames: List[torch.Tensor]):
        self.frames = frames
        self.nb_frames = len(frames)
        self.idx = 0

        assert self.nb_frames > 0, "no frames found in list"

        transforms = v2.Compose([v2.ToPILImage()])
        first_img = transforms(frames[0])
        self.height = first_img.height
        self.width = first_img.width

    def get_resolution(self):
        return self.height, self.width

    def get_frame(self):
        if self.idx >= self.nb_frames:
            return None

        frame = self.frames[self.idx]
        self.idx += 1
        return frame.unsqueeze(0)


class DirectoryReader:
    def __init__(self, dir: str):
        self.paths = sorted(
            glob.glob(os.path.join(dir, "*")),
            key=lambda x: int(os.path.basename(x).split(".")[0]),
        )
        self.nb_frames = len(self.paths)
        self.idx = 0

        assert self.nb_frames > 0, "no frames found in directory"

        first_img = Image.open(self.paths[0])
        self.height = first_img.height
        self.width = first_img.width

    def get_resolution(self):
        return self.height, self.width

    def get_frame(self):
        if self.idx >= self.nb_frames:
            return None

        path = self.paths[self.idx]
        self.idx += 1

        img = Image.open(path)
        transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

        return transforms(img).unsqueeze(0)


class DirectoryWriter:
    def __init__(self, dir: str):
        self.dir = dir
        self.idx = 0

    def open(self):
        return

    def close(self):
        return

    def write_frame(self, frame: torch.Tensor):
        path = f"{self.dir}/{self.idx}.png"
        self.idx += 1

        transforms = v2.Compose([v2.ToPILImage()])
        transforms(frame.squeeze(0)).save(path)


class VideoWriter:
    def __init__(
        self, output_path: str, height: int, width: int, fps: float, format: str
    ):
        self.writer = StreamWriter(dst=output_path)
        self.writer.add_video_stream(
            frame_rate=fps, height=height, width=width, format=format
        )

    def open(self):
        self.writer.open()

    def close(self):
        self.writer.flush()
        self.writer.close()

    def write_frame(self, frame: torch.Tensor):
        self.writer.write_video_chunk(0, frame)
