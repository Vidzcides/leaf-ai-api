import os
import urllib.request
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

DEVICE = "cpu"

# Download SAM model automatically if missing
if not os.path.exists(SAM_CHECKPOINT):
    print("Downloading SAM ViT-B model...")
    urllib.request.urlretrieve(SAM_URL, SAM_CHECKPOINT)
    print("SAM ViT-B downloaded.")

# Load SAM model
sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=DEVICE)

# Lower-memory SAM settings for Railway
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=8,
    points_per_batch=16,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.92,
    crop_n_layers=0,
    min_mask_region_area=500,
)
IMAGE_SIZE = (320, 320)
MIN_CONTOUR_AREA = 5000
MAX_LEAVES = 1  # Only take largest leaf

def extract_leaf(image):

    # Resize large images before SAM to reduce memory usage
    h, w = image.shape[:2]
    max_dim = 768

    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Generate masks
    masks = mask_generator.generate(image_rgb)
    print("SAM masks generated:", len(masks))

    if not masks:
        return cv2.resize(image, IMAGE_SIZE)

    # Take largest masks
    masks = sorted(masks, key=lambda m: m["segmentation"].sum(), reverse=True)[:MAX_LEAVES]

    for mask in masks:

        leaf_mask = np.ascontiguousarray(mask["segmentation"].astype(np.uint8))

        if leaf_mask.sum() < MIN_CONTOUR_AREA:
            continue

        # Find contours
        contours, _ = cv2.findContours(
            leaf_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        c = max(contours, key=cv2.contourArea)

        if cv2.contourArea(c) < MIN_CONTOUR_AREA:
            continue

        clean_mask = np.zeros_like(leaf_mask, dtype=np.uint8)
        cv2.drawContours(clean_mask, [c], -1, 1, thickness=-1)

        # Rotate leaf upright
        rect = cv2.minAreaRect(c)
        angle = rect[2]

        if rect[1][0] < rect[1][1]:
            angle += 90

        h_mask, w_mask = clean_mask.shape
        center = (w_mask // 2, h_mask // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated_image = cv2.warpAffine(
            image_rgb,
            M,
            (w_mask, h_mask),
            flags=cv2.INTER_CUBIC,
            borderValue=(0, 0, 0)
        )

        rotated_mask = cv2.warpAffine(
            clean_mask,
            M,
            (w_mask, h_mask),
            flags=cv2.INTER_NEAREST,
            borderValue=0
        )

        rotated_mask = (rotated_mask > 0).astype(np.uint8)

        # Morph cleanup
        kernel = np.ones((3, 3), np.uint8)

        rotated_mask = cv2.morphologyEx(
            rotated_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        # Keep largest connected component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            rotated_mask,
            connectivity=8
        )

        if num_labels <= 1:
            continue

        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

        rotated_mask = (labels == largest_label).astype(np.uint8)

        if rotated_mask.sum() < MIN_CONTOUR_AREA:
            continue

        # Tight crop
        contours, _ = cv2.findContours(
            rotated_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        c = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(c)

        cropped_leaf = rotated_image[y:y + h, x:x + w]
        cropped_mask = rotated_mask[y:y + h, x:x + w]

        # Black background
        black_bg = np.zeros_like(cropped_leaf)

        black_bg[cropped_mask != 0] = cropped_leaf[cropped_mask != 0]

        # Resize + pad
        h_crop, w_crop = black_bg.shape[:2]

        scale = min(
            IMAGE_SIZE[0] / h_crop,
            IMAGE_SIZE[1] / w_crop
        )

        new_w = int(w_crop * scale)
        new_h = int(h_crop * scale)

        resized_leaf = cv2.resize(
            black_bg,
            (new_w, new_h),
            interpolation=cv2.INTER_LANCZOS4
        )

        final_leaf = np.zeros(
            (IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
            dtype=np.uint8
        )

        x_offset = (IMAGE_SIZE[1] - new_w) // 2
        y_offset = (IMAGE_SIZE[0] - new_h) // 2

        final_leaf[
            y_offset:y_offset + new_h,
            x_offset:x_offset + new_w
        ] = resized_leaf

        return cv2.cvtColor(final_leaf, cv2.COLOR_RGB2BGR)

    # fallback
    return cv2.resize(image, IMAGE_SIZE)