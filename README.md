# A Fair Ranking and New Model for Panoptic Scene Graph Generation

This is the official implementation of our papers

- "DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime", accepted at CVPR 2026. For more information, please visit the [project page](https://lorjul.github.io/dsflash/).
- "A Fair Ranking and New Model for Panoptic Scene Graph Generation", accepted at ECCV 2024. For more information, please visit the [project page](https://lorjul.github.io/fair-psgg/).

## Setup

### Environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). In the folder that contains this README, run `uv sync`, then activate the environment using `source .venv/bin/activate`.

Alternatively, you can use pip to install the different packages listed in the `pyproject.toml` file manually. However, using uv makes things a lot easier!

### Data

<!-- direct link: https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/EgQzvsYo3t9BpxgMZ6VHaEMBDAb7v0UgI8iIAExQUJq62Q?e=fIY3zh -->

Download the PSG datasets from [here](https://github.com/Jingkang50/OpenPSG?tab=readme-ov-file#updates) and extract them.

## Usage

Each of the listed scripts in this section have a `--help` option for more information.

### Training

To train a model, use the following code:

``` sh
python -m fair_psgg \
    --anno /path/to/openpsg/psg/psg.json \
    --img /path/to/openpsg/coco \
    --seg /path/to/openpsg/coco \
    --epochs 40 \
    ./configs/table2/masks-loc-sem.json \
    /output/path/masks-loc-sem
```

For more information, run `python -m fair_psgg --help`.

The provided config files are grouped by the supplementary tables.

### Inference (PredCls)

Run the following to produce an output file using a trained DSFormer model:

``` sh
python -m fair_psgg.tasks.inference \
    /path/to/openpsg/psg/psg.json \
    /path/to/openpsg/coco \
    /path/to/openpsg/coco \
    /path/to/model/output/folder \
    /tmp/results-file.pkl \
    --split test
```

Note that this script does not evaluate the model, but only produces an output file. To evaluate that output file, use the following script:

``` sh
python scripts/evaluate.py \
    /path/to/openpsg/psg/psg.json \
    /tmp/results-file.pkl \
    /tmp/metrics.csv \
    --seg /path/to/openpsg/coco \
    --dedup fail
```

### Inference Without Ground Truth Masks

We show that a good segmentation model improves performance considerably. To reproduce our results, you first need to obtain the inferred segmentation masks.

In this section, we use the following shell variables for better readability:

``` sh
# the path to the folder that contains the psg and coco folders
OPENPSG_DIR=/path/to/openpsg
# output path, where inferred segmentation masks will be written to by a segmentation model
INFMASK_DIR=/path/to/inferred_masks
```

``` sh
# example for OneFormer
python scripts/infer_segmasks.py \
    "$OPENPSG_DIR/coco/val2017" \
    "$INFMASK_DIR/val2017" \
    "$OPENPSG_DIR/psg/psg.json" \
    --model shi-labs/oneformer_coco_swin_large
```

This will write segmentation masks for each image in the provided image directory (first command line argument).

We have testsed the following values for the `--model` option:

- shi-labs/oneformer_coco_swin_large
- facebook/maskformer-swin-tiny-coco
- facebook/maskformer-swin-large-coco
- facebook/mask2former-swin-large-coco-panoptic

Next, match the generated output masks to the PSG annotations:

``` sh
python scripts/match_psg.py --enforce-rel \
    "$OPENPSG_DIR/psg/psg.json" \
    "$INFMASK_DIR/val2017/anno.json" \
    "$INFMASK_DIR/val2017/matched.json"
```

If you are interested, you can show some stats about the matching by running the following:

``` sh
python scripts/get_match_stats.py \
    "$OPENPSG_DIR/psg/psg.json" \
    "$INFMASK_DIR/val2017/matched.json"
```

Now, you have a new annotation file that can be used with DSFormer. Run inference as shown above in the Usage section. DSFormer will produce a results file that can be evaluated using the following script:

``` sh
# run inference
# replace /path/to/trained/model/folder with an actual path to a trained DSFormer model folder
# DSFormer model folders usually contain a config.json and a last_state.pth
python -m fair_psgg.tasks.inference \
    "$INFMASK_DIR/val2017/matched.json" \
    "$OPENPSG_DIR/coco" \
    "$INFMASK_DIR" \
    /path/to/trained/model/folder \
    /tmp/results-file.pkl \
    --workers 12 \
    --split test

# evaluate results file
python scripts/evaluate.py \
    "$OPENPSG_DIR/psg/psg.json" \
    /tmp/results-file.pkl \
    /tmp/metrics.csv \
    --seg "$OPENPSG_DIR/coco" \
    --dedup fail
```

## Citation

If you found our paper or code helpful, please consider citing it:

``` bibtex
@misc{lorenz2024fairpsgg,
    title={A Fair Ranking and New Model for Panoptic Scene Graph Generation}, 
    author={Julian Lorenz and Alexander Pest and Daniel Kienzle and Katja Ludwig and Rainer Lienhart},
    year={2024},
    eprint={2407.09216},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2407.09216}, 
}
```
