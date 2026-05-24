# Project: PDS 2026 - Skin Lesion Classification - Group Quokka
# Hair and marker removal from skin lesion images.

import os
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import morphology, restoration, util


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

img_dir   = os.path.join(ROOT, "data", "imgs")
mask_dir  = os.path.join(ROOT, "data", "masks")
out_dir   = os.path.join(ROOT, "data", "imgs_clean")

timeout_sec = 20


def hair_mask(gray, lesion_mask):
    """Mask for hair: aggressive on the skin, gentle on the lesion."""
    blackhat = morphology.black_tophat(gray, morphology.disk(5))
    tophat   = morphology.white_tophat(gray, morphology.disk(5))

    # Skin uses 0.04, lesion uses stricter 0.10 (only strong hair on the lesion)
    threshold = np.where(lesion_mask > 0, 0.10, 0.04)
    mask = (blackhat > threshold) | (tophat > threshold)
    return morphology.dilation(mask, morphology.disk(1))


def marker_mask(gray, lesion_mask):
    """Mask for thick dark marker ink lines, but only outside the lesion."""
    blackhat = morphology.black_tophat(gray, morphology.disk(25))
    mask = blackhat > 0.04
    mask = morphology.dilation(mask, morphology.disk(8))
    mask[lesion_mask > 0] = False
    return mask


def inpaint(img, mask):
    """Fill the masked pixels by interpolating from surrounding pixels."""
    if not mask.any():
        return util.img_as_float(img)
    return restoration.inpaint_biharmonic(util.img_as_float(img), mask, channel_axis=-1)


def clean_image(img, lesion_mask, out_path):
    """Build hair + marker masks together, inpaint once, and save to disk."""
    gray = rgb2gray(img)
    mask = hair_mask(gray, lesion_mask) | marker_mask(gray, lesion_mask)
    plt.imsave(out_path, inpaint(img, mask))


def process_folder(input_folder, mask_folder, output_folder):
    """Clean every image in the folder and save with a 'clean_' prefix."""
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.endswith(".png")]

    for i, filename in enumerate(files):
        print(f"[{i + 1}/{len(files)}] {filename}")

        out_path = os.path.join(output_folder, "clean_" + filename)
        if os.path.exists(out_path):
            print("  skipped (already done)")
            continue

        mask_path = os.path.join(mask_folder, filename.replace(".png", "_mask.png"))
        if not os.path.exists(mask_path):
            print("  skipped (no mask)")
            continue

        img = plt.imread(os.path.join(input_folder, filename))
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        lesion_mask = plt.imread(mask_path)
        if lesion_mask.ndim == 3:
            lesion_mask = rgb2gray(lesion_mask)
        lesion_mask = lesion_mask > 0.5

        p = mp.Process(target=clean_image, args=(img, lesion_mask, out_path))
        p.start()
        p.join(timeout_sec)
        if p.is_alive():
            p.terminate()
            p.join()
            print(f"  skipped (stuck > {timeout_sec}s)")


if __name__ == "__main__":
    process_folder(img_dir, mask_dir, out_dir)
