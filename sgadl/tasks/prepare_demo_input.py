# set category_id to 0 for all annotations
# remove ignored instances
# create better colours for the segmentation map
# rename file_name to img, and pan_seg_file_name to seg

from argparse import ArgumentParser
from pathlib import Path
import json
import pickle
import shutil
import numpy as np
from PIL import Image
from sgadl.utils import id2rgb, open_segmask


def cli():
    parser = ArgumentParser()
    parser.add_argument("anno")
    parser.add_argument("search")
    parser.add_argument("output")
    parser.add_argument("--ignore", default=None)
    args = parser.parse_args()

    img_dir = Path(args.anno).parent
    with open(args.anno, "rb") as f:
        anno = pickle.load(f)

    ignore_nodes = set()
    if args.ignore:
        with open(args.ignore) as f:
            config = json.load(f)
        for n in config["data"]["ignore_nodes"]:
            ignore_nodes.add(anno["node_names"].index(n))
        print("Ignoring", len(ignore_nodes))

    for x in anno["data"]:
        if x["file_name"] == args.search:
            to_write = dict(x)
            fns = to_write.pop("file_name").split("/")
            to_write["img"] = fns[0] + "_" + fns[-1]
            pns = to_write.pop("pan_seg_file_name").split("/")
            to_write["seg"] = pns[0] + "_" + pns[-1]
            old_seg = open_segmask(img_dir / x["pan_seg_file_name"])
            new_seg = np.zeros_like(old_seg)
            new_seginfo = []
            new_anno = []
            for a, s, new_id in zip(
                to_write["annotations"],
                to_write["segments_info"],
                np.linspace(0, 255 * 255 * 255, len(to_write["segments_info"]) + 1)
                .round()
                .astype(int)
                .tolist()[1:],
            ):
                new_seg[old_seg == s["id"]] = new_id
                if a["category_id"] not in ignore_nodes:
                    a.pop("category_id")
                    new_anno.append(a)
                    new_seginfo.append({"id": new_id})
            to_write["annotations"] = new_anno
            to_write["segments_info"] = new_seginfo
            to_write.pop("param_relations")
            outdir = Path(args.output)
            outdir.mkdir(exist_ok=True)
            with open(outdir / "input.json", "w") as f:
                json.dump(to_write, f, indent=2)

            shutil.copy(img_dir / x["file_name"], outdir / to_write["img"])
            Image.fromarray(id2rgb(new_seg)).save(outdir / to_write["seg"])
            break
    else:
        print("Entry not found")


if __name__ == "__main__":
    cli()
