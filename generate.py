import asyncio
import base64
from datetime import datetime
import httpx
import os

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SD_API_URL = "https://mlkyway.anselbrandt.net/sdapi/v1/txt2img"  # Update if local


def timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def decode_and_save_base64(image_b64: str, file_path: str):
    image_data = base64.b64decode(image_b64)
    with open(file_path, "wb") as f:
        f.write(image_data)
    print(f"✅ Image saved to {file_path}")


async def generate_image_from_prompt(prompt: str, negative_prompt: str) -> str:
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

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(SD_API_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            image_b64 = data["images"][0]
            file_path = os.path.join(OUTPUT_DIR, f"txt2img-{timestamp()}.png")
            decode_and_save_base64(image_b64, file_path)
            return file_path
        except httpx.ConnectTimeout:
            print(
                "❌ Connection timed out! Check if your server is running and accessible."
            )
        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")


async def main():
    prompt = (
        "A photorealistic image of a young girl, around 4 years old, with light skin and slightly messy shoulder-length brown hair, standing close to the camera in the foreground."
        "She is giving a subtle, mischievous smirk while glancing sideways at the viewer."
        "In the background, a suburban house is fully engulfed in flames, with bright orange fire and thick black smoke rising into the overcast sky."
        "Firefighters and bystanders stand near the house."
        "A large white fire truck is present with firefighters working on it."
        "Yellow fire hoses are stretched across a residential street."
        "The scene is framed like an early 2000s candid photo, with a shallow depth of field that keeps the girl in sharp focus while the fire and firefighters are slightly blurred."
    )
    negative_prompt = "older girl, centered pose, hands on hips, modern fire truck only, extra limbs, cartoonish style, low resolution, fantasy elements, futuristic elements, text overlay, watermark"
    print(f"📝 Generated prompt:\n{prompt}\n")

    image_path = await generate_image_from_prompt(prompt, negative_prompt)
    if image_path:
        print(f"✅ Image generated at {image_path}")
    else:
        print("⚠️ No image generated.")


if __name__ == "__main__":
    asyncio.run(main())
