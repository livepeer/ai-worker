from cog import BasePredictor, Input, Path
from pipelines.svd_film import StableVideoDiffusionFILMPipeline
import os


class Predictor(BasePredictor):
    def setup(self):
        self.pipeline = StableVideoDiffusionFILMPipeline(cache_dir="./cache")

    def predict(
        self,
        image: Path = Input(description="Image to guide generation"),
        motion_bucket_id: float = Input(
            description="The motion bucket ID. Used as conditioning for the generation. The higher the number the more motion will be in the video",
            default=127,
        ),
        noise_aug_strength: float = Input(
            description="The amount of noise added to the init image, the higher it is the less the video will look like the init image. Increase it for more motion",
            default=0.02,
        ),
    ) -> Path:
        output_path = os.path.join("output", "output.mp4")
        self.pipeline(
            output_path=output_path,
            image=image,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
        )

        # TODO: Ensure that these temp files are cleared
        return Path(output_path)
