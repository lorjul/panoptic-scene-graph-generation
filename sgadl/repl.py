import json
from pathlib import Path
from PIL import Image
from project_paths import project_paths

STD_RELNAMES = (
    "i_behind",
    "i_front",
    "i_right",
    "i_left",
    "i_above",
    "i_below",
    "d_behind",
    "d_front",
    "d_right",
    "d_left",
    "i_distance",
    "i_touching",
    "i_on",
)


def load_psg_json():
    with open(project_paths.psg_annotation_dir) as f:
        return json.load(f)


def load_psg_image(item):
    if isinstance(item, str):
        file_name = item
    elif isinstance(item, dict):
        file_name = item["file_name"]
    else:
        raise RuntimeError()
    return Image.open(Path(project_paths.psg_img_dir) / file_name).convert("RGB")
