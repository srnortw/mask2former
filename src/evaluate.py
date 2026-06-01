import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as coco_mask
import json, os, tempfile


def evaluate(model, val_loader, device, score_threshold: float = 0.5):
    """Run COCO evaluation and return mask AP."""
    model.eval()
    results = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)

            outputs = model(pixel_values=images)

            # Post-process predictions
            pred_masks  = outputs.masks_queries_logits.sigmoid()   # [B, Q, H, W]
            pred_logits = outputs.class_queries_logits              # [B, Q, num_classes+1]

            pred_scores = pred_logits.softmax(-1)[:, :, :-1].max(-1)  # exclude no-object
            pred_labels = pred_logits.softmax(-1)[:, :, :-1].argmax(-1)

            for b in range(images.shape[0]):
                img_id = targets[b]["image_id"].item()
                scores = pred_scores.values[b]
                labels = pred_labels[b]
                masks  = pred_masks[b]

                keep = (scores > score_threshold).nonzero(as_tuple=True)[0]
                for q in keep:
                    mask = (masks[q] > 0.5).cpu().numpy().astype(np.uint8)
                    rle  = coco_mask.encode(np.asfortranarray(mask))
                    rle["counts"] = rle["counts"].decode("utf-8")

                    results.append({
                        "image_id":     img_id,
                        "category_id":  labels[q].item() + 1,  # COCO is 1-indexed
                        "segmentation": rle,
                        "score":        scores.values[q].item(),
                    })

    if not results:
        return 0.0

    # Write predictions to temp file for COCOeval
    gt_file  = val_loader.dataset.ann_file
    coco_gt  = COCO(gt_file)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(results, f)
        pred_file = f.name

    coco_dt   = coco_gt.loadRes(pred_file)
    evaluator = COCOeval(coco_gt, coco_dt, iouType="segm")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    os.unlink(pred_file)
    return float(evaluator.stats[0])   # AP @ IoU 0.50:0.95
