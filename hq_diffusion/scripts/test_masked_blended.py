import diffusers
import torch
from diffusers import StableDiffusion3Pipeline, StableDiffusion3InpaintPipeline
import sys
from diffusers.utils import load_image
import numpy as np

if __name__ == "__main__":

    input_model = sys.argv[1]
    # pipe = StableDiffusion3Pipeline.from_single_file(input_model, torch_dtype=torch.bfloat16)
    pipe = StableDiffusion3InpaintPipeline.from_single_file(input_model, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")

    init_image = diffusers.utils.load_image("/root/autodl-tmp/tmp.jpg").convert("RGB")
    mask_image = torch.zeros(init_image.size[1], init_image.size[0], 3)  # All white mask for "Only Masked" behavior
    mask_image[0:100, 0:100, :] = 1  # Small black square in top-left corner
    # # make mask_image to pil image

    mask_image = diffusers.utils.numpy_to_pil(mask_image.numpy())[0]
    # init_image = diffusers.utils.load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
    # mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")

    result = pipe(
        prompt="a human face",
        image=init_image,
        mask_image=mask_image,
        padding_mask_crop=32,  # Crucial for "Only Masked" behavior
        # num_inference_steps=40,
        strength=0.8,
    ).images[0]

    # image = pipe(
    #     "a cat in grasslands",
    #     num_inference_steps=40,
    #     guidance_scale=4.5,
    # ).images[0]
    result.save("/root/autodl-tmp/tmp2.jpg")
    pass