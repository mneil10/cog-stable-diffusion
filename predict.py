import os
import requests
from typing import List


import torch
print(torch.__version__)
from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionPipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)

# Model configuration
S3_URL = "https://my-models-bucket12.s3.eu-west-2.amazonaws.com/model.safetensors?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMD%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCWV1LXdlc3QtMiJHMEUCIQDJhZs1mwQGwUpTX7gEAQ4JSuREZDT4%2FDdjLVMLVum2LQIgXG56B2FGiwNI5reVb%2Bh%2FwI2XfpKCJQMB%2BCCdf1P1UDgqxwMIGRAAGgw0MjI3MTg5MzMwNzQiDNGWehYwnsgHKrycZyqkAwBNwuEEuKZk61VdR7jYPin5qkKlaixhdqtZrbppqR0tKm7rXJI9abS5um5a%2FN4R0luo7L9dDXKqugOM2geeyKVTeFiOz%2BZzsQcOd6tg7HJU93FtbK%2BY43FRllMMavWNWhBRZeiz%2FfejPKdu%2F82hpNAIJVEtMzsYMTZ0jClNVvVVxVOcTUlJkjOfgZaz3n7lDf%2BFLaNLW9ZnA82kNEW3kD7AiHAx55AM3ub2u0as6v%2FVUa2wkOZmjnPtCwWyM7iskqgH1SF0Oi%2BLX6aKsNdzgAXjcO12vD1LHF7Qm4fKHKnhVIkcDELvy0L%2FjNBd2fqf5q1FfteWg4Eik6Gy49I6rg%2FC4OAq3C%2Fm9kcqBQQybHmHwM4w8HfLgXrdNVDU%2Brl%2BfTrLpemvhj%2FLLxMF3NWOSSS800%2Fg5TuGbySWKiSDSUbPTKFO1trHtXdNqgCmtUlowkaLzjifT1QJ4xcNcNY%2BKCP3r8gYdCP%2B1sPimKI6ijBRgWgRL8rK2onLUGG1z4bhqml3XmYORI22oB4L%2BxgUxuq2DZeD12gUSIcf1VaaC06uIzgQDTDUxta%2BBjrkAl3l2PoOMKJCeJtjPkP8pC8acL%2B5DYYjgK5oszhFO40nrgs83xPf2n4uB6j9NDLT7%2BZIwbfeTMCsmW%2BxVJIeanWBQhmhTRm%2FCj8MRSDRD%2BsbasQ4kAu%2Bz%2FketRt0I8Yrevhy%2Fgzyp8ZBuB3mE93DbREfzWSXcl%2BU26zqIUnIbc%2FzDFIzrjJ%2F8H6M5JOE%2BE6kBATNA2Lf8Xb%2Bfiq9p5q6RTpmQO%2FrdtOygFJ%2FI%2BY9z0JnmIq5%2FEMc2mg9Nq3KeKlQmSFXTTHHRy1yTo%2FG2%2B%2FtGvI85z6AOFo2G4ydx%2F43yFDn5ohgShtdDYZ55VJO0qs%2FiIHn%2Fk5dmFPN9FQsYLY8tWRR15k4Nb21IhKFJTjsQT%2FKKsI4gjo1CgsPimoKYCHbjkCx5KAvfvYXBpzzhdGhDHn36qYVvTLSOkyLWbTF95WD3yMlfTuhx3ZzFUfN8TwB6Ly6HsD9OvsiUxFfqZ68aCiCZRhW&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAWE3ADBBJJSNGJPCM%2F20250315%2Feu-west-2%2Fs3%2Faws4_request&X-Amz-Date=20250315T155752Z&X-Amz-Expires=10800&X-Amz-SignedHeaders=host&X-Amz-Signature=115324c90ba261c1d682c9fe97c0e7265fdd2452befaa76d58be10b1bff741c8"
LOCAL_PATH = "/tmp/model.safetensors"
SAFETY_MODEL_ID = "CompVis/stable-diffusion-safety-checker"
MODEL_CACHE = "diffusers-cache"

def download_model():
    if not os.path.exists(LOCAL_PATH):
        print("Downloading model from S3...")
        response = requests.get(S3_URL, stream=True)
        with open(LOCAL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    return LOCAL_PATH

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_MODEL_ID,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        )
        
        # Load custom model from S3
        model_path = download_model()
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            safety_checker=safety_checker,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to("cuda")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="a photo of an astronaut riding a horse on mars",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default=None,
        ),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=768,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=768,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        scheduler: str = Input(
            default="DPMSolverMultistep",
            choices=[
                "DDIM",
                "K_EULER",
                "DPMSolverMultistep",
                "K_EULER_ANCESTRAL",
                "PNDM",
                "KLMS",
            ],
            description="Choose a scheduler.",
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        self.pipe.scheduler = make_scheduler(scheduler, self.pipe.scheduler.config)

        generator = torch.Generator("cuda").manual_seed(seed)
        output = self.pipe(
            prompt=[prompt] * num_outputs if prompt is not None else None,
            negative_prompt=[negative_prompt] * num_outputs
            if negative_prompt is not None
            else None,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
        )

        output_paths = []
        for i, sample in enumerate(output.images):
            if output.nsfw_content_detected and output.nsfw_content_detected[i]:
                continue

            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]
