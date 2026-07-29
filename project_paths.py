import socket
from pathlib import Path


class PathContainer:
    psg_img_dir: Path
    psg_seg_dir: Path
    psg_annotation_dir: Path
    vg_img_dir: Path
    vg_annotation_dir: Path
    eomt_weights: Path
    dinov3_weights: Path

    def __init__(self, **kwargs):
        self.paths = kwargs

    def __getattr__(self, item):
        p = self.paths.get(item, None)
        if p is None:
            raise KeyError(f"Project path not set for: {item}")
        return Path(p)

    def get_img_dir(self, dataset: str):
        if dataset == "psg":
            return self.psg_img_dir
        if dataset == "vg-kaihua":
            return self.vg_img_dir
        return None

    def get_seg_dir(self, dataset: str):
        if dataset == "psg":
            return self.psg_seg_dir
        return None

    def get_anno_path(self, dataset: str):
        if dataset == "psg":
            return self.psg_annotation_dir
        if dataset == "vg-kaihua":
            return self.vg_annotation_dir
        return None


project_paths = PathContainer()

current_host_name = socket.gethostname()
if current_host_name == "yourhost":
    project_paths = PathContainer(
        psg_img_dir="/data/psg/coco",
        psg_seg_dir="/data/psg/coco",
        psg_annotation_dir="/data/psg/psg/psg.json",
        eomt_weights="/data/weights/eomt",
        dinov3_weights="/data/weights/dinov3",
    )
