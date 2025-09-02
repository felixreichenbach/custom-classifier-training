import asyncio
from datetime import datetime
import os
from pathlib import Path

from dotenv import load_dotenv
from viam.rpc.dial import DialOptions
from viam.app.viam_client import ViamClient

# ---------- config ----------
load_dotenv()

API_KEY = os.environ["VIAM_API_KEY"]
API_KEY_ID = os.environ["VIAM_API_KEY_ID"]
PART_ID = os.environ["PART_ID"]  # required for upload to associate image w part
DATA_SET_ID = os.environ["DATA_SET_ID"]
SCRIPT_DIR = Path(__file__).parent
IMAGES_DIR = (
    SCRIPT_DIR / "dataset/data_cropped_4_3"
)  # Change this to your images folder
TAGS = ["NOK"]
# --------------------------------


async def connect() -> ViamClient:
    dial_options = DialOptions.with_api_key(api_key=API_KEY, api_key_id=API_KEY_ID)
    return await ViamClient.create_from_dial_options(dial_options)


def get_image_files(directory: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    return [f for f in directory.iterdir() if f.suffix.lower() in exts and f.is_file()]


async def main():
    client = await connect()
    try:
        image_files = get_image_files(IMAGES_DIR)
        if not image_files:
            print(f"No images found in {IMAGES_DIR}")
            return
        for img_path in image_files:
            with open(img_path, "rb") as img_file:
                img_bytes = img_file.read()
            time_requested = time_received = datetime.now()
            fid = await client.data_client.binary_data_capture_upload(
                part_id=PART_ID,
                tags=TAGS,
                component_type="camera",
                component_name="my_camera",
                method_name="GetImages",
                method_parameters=None,
                data_request_times=[time_requested, time_received],
                file_extension=".jpg",
                binary_data=img_bytes,
                dataset_ids=[DATA_SET_ID],
            )
            print(f"✅ uploaded {img_path.name} as {fid}")
    finally:
        client.close()


if __name__ == "__main__":
    asyncio.run(main())
