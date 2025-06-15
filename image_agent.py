"""An agent that generates images using Stable Diffusion based on OpenAI's prompt suggestions.

This agent takes a request to generate an image similar to the disaster girl meme,
uses OpenAI to generate appropriate prompts, and then uses Stable Diffusion to create the image.
"""

from __future__ import annotations as _annotations

import asyncio
import base64
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Any

import httpx
import logfire
from devtools import debug
from dotenv import load_dotenv

from pydantic_ai import Agent, ModelRetry, RunContext

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SD_API_URL = "https://mlkyway.anselbrandt.net/sdapi/v1/txt2img"

logfire.configure()
logfire.instrument_httpx()
logfire.instrument_pydantic_ai()

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@dataclass
class Deps:
    client: httpx.AsyncClient


def timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def decode_and_save_base64(image_b64: str, file_path: str):
    image_data = base64.b64decode(image_b64)
    with open(file_path, "wb") as f:
        f.write(image_data)
    print(f"✅ Image saved to {file_path}")


image_agent = Agent(
    "openai:gpt-4",
    instructions=(
        "You are an expert at creating detailed prompts for Stable Diffusion that will generate "
        "images similar to the disaster girl meme. The disaster girl meme features a young girl "
        "with a mischievous smirk while a house burns in the background. "
        "Generate both a positive prompt that describes the desired image in detail and a negative "
        "prompt that specifies what to avoid. Be very specific about the composition, lighting, "
        "and style to match the original meme's aesthetic."
    ),
    deps_type=Deps,
    retries=2,
)


@image_agent.tool
async def generate_image(
    ctx: RunContext[Deps], prompt: str, negative_prompt: str
) -> dict[str, Any]:
    """Generate an image using Stable Diffusion.

    Args:
        ctx: The context.
        prompt: The positive prompt describing what to generate.
        negative_prompt: The negative prompt describing what to avoid.
    """
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": 30,
        "cfg_scale": 7,
        "width": 1600,
        "height": 1080,
        "sampler_index": "Euler",
        "batch_size": 1,
        "n_iter": 1,
        "restore_faces": False,
        "tiling": False,
        "send_images": True,
        "save_images": False,
    }

    try:
        response = await ctx.deps.client.post(SD_API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        image_b64 = data["images"][0]
        file_path = os.path.join(OUTPUT_DIR, f"disaster-girl-{timestamp()}.png")
        decode_and_save_base64(image_b64, file_path)
        return {"file_path": file_path}
    except httpx.ConnectTimeout:
        raise ModelRetry("Connection timed out! Check if the Stable Diffusion server is running.")
    except httpx.HTTPStatusError as e:
        raise ModelRetry(f"HTTP error {e.response.status_code}: {e.response.text}")
    except Exception as e:
        raise ModelRetry(f"Unexpected error: {e}")


async def main():
    async with httpx.AsyncClient(timeout=30.0) as client:
        logfire.instrument_httpx(client, capture_all=True)
        deps = Deps(client=client)
        result = await image_agent.run(
            "Generate an image similar to the disaster girl meme using Stable Diffusion.",
            deps=deps,
        )
        debug(result)
        print("Response:", result.output)


if __name__ == "__main__":
    asyncio.run(main())
