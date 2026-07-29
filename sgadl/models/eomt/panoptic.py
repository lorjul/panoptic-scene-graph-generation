import numpy as np
import torch
from PIL import Image
from torch.nn.functional import interpolate
from torchvision.transforms.v2.functional import pad


def scale_img_size_instance_panoptic(model_img_size, size: tuple[int, int]):
    factor = min(
        model_img_size[0] / size[0],
        model_img_size[1] / size[1],
    )

    return [round(s * factor) for s in size]


@torch.compiler.disable
def resize_and_pad_imgs_instance_panoptic(model_img_size, imgs):
    transformed_imgs = []

    for img in imgs:
        new_h, new_w = scale_img_size_instance_panoptic(model_img_size, img.shape[-2:])

        pil_img = Image.fromarray(img.permute(1, 2, 0).cpu().numpy())
        pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
        resized_img = (
            torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).to(img.device)
        )

        pad_h = max(0, model_img_size[-2] - resized_img.shape[-2])
        pad_w = max(0, model_img_size[-1] - resized_img.shape[-1])
        padding = [0, 0, pad_w, pad_h]

        padded_img = pad(resized_img, padding)

        transformed_imgs.append(padded_img)

    return torch.stack(transformed_imgs)


@torch.compiler.disable
def revert_resize_and_pad_logits_instance_panoptic(
    model_img_size, transformed_logits, img_sizes
):
    logits = []
    for i in range(len(transformed_logits)):
        scaled_size = scale_img_size_instance_panoptic(model_img_size, img_sizes[i])
        logits_i = transformed_logits[i][:, : scaled_size[0], : scaled_size[1]]
        logits_i = interpolate(
            logits_i[None, ...],
            img_sizes[i],
            mode="bilinear",
        )[0]
        logits.append(logits_i)

    return logits


def to_per_pixel_preds_panoptic(
    num_classes,
    mask_logits_list,
    class_logits,
    stuff_classes,
    mask_thresh,
    overlap_thresh,
):
    scores, classes = class_logits.softmax(dim=-1).max(-1)
    preds_list = []

    for i in range(len(mask_logits_list)):
        preds = -torch.ones(
            (*mask_logits_list[i].shape[-2:], 2),
            dtype=torch.long,
            device=class_logits.device,
        )
        preds[:, :, 0] = num_classes

        keep = classes[i].ne(class_logits.shape[-1] - 1) & (scores[i] > mask_thresh)
        if not keep.any():
            preds_list.append(preds)
            continue

        masks = mask_logits_list[i].sigmoid()
        segments = -torch.ones(
            *masks.shape[-2:],
            dtype=torch.long,
            device=class_logits.device,
        )

        mask_ids = (scores[i][keep][..., None, None] * masks[keep]).argmax(0)
        stuff_segment_ids, segment_id = {}, 0
        segment_and_class_ids = []

        for k, class_id in enumerate(classes[i][keep].tolist()):
            orig_mask = masks[keep][k] >= 0.5
            new_mask = mask_ids == k
            final_mask = orig_mask & new_mask

            orig_area = orig_mask.sum().item()
            new_area = new_mask.sum().item()
            final_area = final_mask.sum().item()
            if (
                orig_area == 0
                or new_area == 0
                or final_area == 0
                or new_area / orig_area < overlap_thresh
            ):
                continue

            if class_id in stuff_classes:
                if class_id in stuff_segment_ids:
                    segments[final_mask] = stuff_segment_ids[class_id]
                    continue
                else:
                    stuff_segment_ids[class_id] = segment_id

            segments[final_mask] = segment_id
            segment_and_class_ids.append((segment_id, class_id))

            segment_id += 1

        for segment_id, class_id in segment_and_class_ids:
            segment_mask = segments == segment_id
            preds[:, :, 0] = torch.where(segment_mask, class_id, preds[:, :, 0])
            preds[:, :, 1] = torch.where(segment_mask, segment_id, preds[:, :, 1])

        preds_list.append(preds)

    return preds_list
