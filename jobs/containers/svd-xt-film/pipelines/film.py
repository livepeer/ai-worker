import torch
from torchvision.transforms import v2
from tqdm import tqdm
import bisect
import numpy as np


class FILMPipeline:
    model: torch.jit.ScriptModule

    def __init__(self, model_path: str):
        self.model = torch.jit.load(model_path, map_location="cpu")
        self.model.eval()

    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        return self

    @property
    def device(self) -> torch.device:
        # Checking device for ScriptModule requires checking one of its parameters
        params = self.model.parameters()
        return next(params).device

    @property
    def dtype(self) -> torch.dtype:
        # Checking device for ScriptModule requires checking one of its parameters
        params = self.model.parameters()
        return next(params).dtype

    def __call__(
        self,
        reader,
        writer,
        inter_frames: int = 2,
    ):
        transforms = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
            ]
        )

        writer.open()

        while True:
            frame_1 = reader.get_frame()
            # If the first frame read is None then there are no more frames
            if frame_1 is None:
                break

            frame_2 = reader.get_frame()
            # If the second frame read is None there there is a final frame
            if frame_2 is None:
                writer.write_frame(transforms(frame_1))
                break

            # frame_1 and frame_2 must be tensors with n c h w format
            frame_1 = frame_1.unsqueeze(0)
            frame_2 = frame_2.unsqueeze(0)

            frames = inference(
                self.model, frame_1, frame_2, inter_frames, self.device, self.dtype
            )

            frames = [transforms(frame.detach().cpu()) for frame in frames]
            for frame in frames:
                writer.write_frame(frame)

        writer.close()


def inference(
    model, img_batch_1, img_batch_2, inter_frames, device, dtype
) -> torch.Tensor:
    results = [img_batch_1, img_batch_2]

    idxes = [0, inter_frames + 1]
    remains = list(range(1, inter_frames + 1))

    splits = torch.linspace(0, 1, inter_frames + 2)

    for _ in tqdm(range(len(remains)), "Generating in-between frames"):
        starts = splits[idxes[:-1]]
        ends = splits[idxes[1:]]
        distances = (
            (splits[None, remains] - starts[:, None])
            / (ends[:, None] - starts[:, None])
            - 0.5
        ).abs()
        matrix = torch.argmin(distances).item()
        start_i, step = np.unravel_index(matrix, distances.shape)
        end_i = start_i + 1

        x0 = results[start_i].to(device=device, dtype=dtype)
        x1 = results[end_i].to(device=device, dtype=dtype)

        dt = x0.new_full((1, 1), (splits[remains[step]] - splits[idxes[start_i]])) / (
            splits[idxes[end_i]] - splits[idxes[start_i]]
        )

        with torch.no_grad():
            prediction = model(x0, x1, dt)
        insert_position = bisect.bisect_left(idxes, remains[step])
        idxes.insert(insert_position, remains[step])
        results.insert(insert_position, prediction.clamp(0, 1).float())
        del remains[step]

    return results
