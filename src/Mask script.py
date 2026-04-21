import os
import numpy as np
import matplotlib.pyplot as plt

from skimage.color import rgb2gray
from skimage import morphology
from skimage.filters import gaussian


# ---------------------------
# Load data
# ---------------------------
def load_image(path):
    return plt.imread(path)


def load_mask(path):
    gt = plt.imread(path)

    if gt.max() > 1:
        gt = gt / 255

    return gt.astype(int)


# ---------------------------
# Dice score (optional)
# ---------------------------
def calculate_dice(mask, gt):
    intersection = np.sum(mask * gt)
    return (2 * intersection) / (np.sum(mask) + np.sum(gt) + 1e-8)


# ---------------------------
# Segmentation pipeline
# ---------------------------
def segment_image(im):
    # Convert to grayscale
    gray = rgb2gray(im) * 256

    # Step 1: blur
    blurred = gaussian(gray, sigma=5)

    # Step 2: threshold
    mask = blurred < 120

    # Step 3: morphology
    struct_el = morphology.disk(6)

    mask = morphology.binary_opening(mask, struct_el)
    mask = morphology.binary_closing(mask, struct_el)

    return mask.astype(np.uint8)


# ---------------------------
# Save mask
# ---------------------------
def save_mask(mask, path):
    # Convert to 0–255 image
    plt.imsave(path, mask, cmap='gray')


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":

    img_dir = "../data/imgs"
    mask_dir = "../data/masks"          # optional (for Dice)
    output_dir = "../data/output_masks"

    os.makedirs(output_dir, exist_ok=True)

    dice_scores = []

    for filename in os.listdir(img_dir):

        if not filename.endswith(".png"):
            continue

        image_path = os.path.join(img_dir, filename)

        # Create mask filename
        name, _ = os.path.splitext(filename)
        gt_filename = name + "_mask.png"
        gt_path = os.path.join(mask_dir, gt_filename)

        # Output filename
        output_path = os.path.join(output_dir, gt_filename)

        # Load image
        im = load_image(image_path)

        # Segment
        pred_mask = segment_image(im)

        # Save result
        save_mask(pred_mask, output_path)

        # Evaluate if GT exists
        if os.path.exists(gt_path):
            gt = load_mask(gt_path)
            dice = calculate_dice(pred_mask, gt)
            dice_scores.append(dice)
            print(f"{filename} → Dice: {dice:.4f}")
        else:
            print(f"{filename} → mask saved (no GT found)")

    # Summary
    if dice_scores:
        print("\n--- Summary ---")
        print(f"Average Dice: {np.mean(dice_scores):.4f}")