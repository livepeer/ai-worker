import os
import glob
from PIL import Image
import torch
from torchvision.transforms import v2
from torchaudio.io import StreamWriter

dir = "/home/user/.lpData/offchain/input/640f1aa7d9d9cfea7d1f"

files = sorted(
    glob.glob(os.path.join(dir, "*")),
    key=lambda x: int(os.path.basename(x).split(".")[0]),
)

frames = [Image.open(f) for f in files]

transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.uint8, scale=True)])
frames = transforms(frames)
frames = torch.stack(frames)
frames = frames.to("cuda")

print(frames.shape)

output_path = "test.mp4"
out_config = {"width": 1024, "height": 576, "frame_rate": 7, "format": "rgb0"}
writer = StreamWriter(dst=output_path)
writer.add_video_stream(
    **out_config,
    encoder="h264_nvenc",
    encoder_format="rgb0",
    hw_accel="cuda",
)

with writer.open():
    for frame in frames:
        writer.write_video_chunk(0, frame.unsqueeze(0))
