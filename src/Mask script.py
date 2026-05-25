import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab
from skimage import morphology
from skimage.filters import gaussian, sobel
from skimage.measure import label
from skimage.segmentation import active_contour
import warnings


def load_image(path: str) -> np.ndarray:
    im = plt.imread(path)
    if im.dtype == np.uint8:
        im = im / 255.0
    im = im.astype(np.float64)
    if im.ndim == 3 and im.shape[2] == 4:
        im = im[..., :3]
    return im


def load_mask(path: str) -> np.ndarray:
    gt = plt.imread(path)
    if gt.max() > 1:
        gt = gt / 255.0
    if gt.ndim == 3:
        gt = gt[..., 0]
    return (gt > 0.5).astype(np.uint8)


def calculate_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = np.sum(pred * gt)
    return (2.0 * intersection) / (np.sum(pred) + np.sum(gt) + 1e-8)


def build_edge_map(im: np.ndarray) -> np.ndarray:
    # LAB gives better sensitivity to colour shifts (e.g. brown lesion on
    # pale skin) where the lightness channel alone would miss the boundary.
    lab = rgb2lab(im)
    channels = [
        lab[..., 0] / 100.0,
        (lab[..., 1] + 128) / 255.0,
        (lab[..., 2] + 128) / 255.0,
    ]
    edge_maps = []
    for ch in channels:
        blurred = gaussian(ch, sigma=2)
        edges = sobel(blurred)
        e_min, e_max = edges.min(), edges.max()
        if e_max > e_min:
            edges = (edges - e_min) / (e_max - e_min)
        edge_maps.append(edges)
    return np.mean(edge_maps, axis=0)


def resample_contour(contour: np.ndarray, n_points: int) -> np.ndarray:
    from scipy.interpolate import interp1d
    t_orig = np.linspace(0, 1, len(contour))
    t_new = np.linspace(0, 1, n_points)
    return interp1d(t_orig, contour, axis=0)(t_new)


def extract_blob_contours(mask: np.ndarray, n_points: int = 200) -> list:
    from skimage.measure import find_contours

    labeled = label(mask)
    n_blobs = labeled.max()
    H, W = mask.shape
    contours_out = []

    for blob_id in range(1, n_blobs + 1):
        blob = (labeled == blob_id).astype(np.uint8)
        contours = find_contours(blob, 0.5)
        if not contours:
            continue
        contour = max(contours, key=len)
        if len(contour) < 4:
            contours_out.append(contour)
            continue
        contours_out.append(resample_contour(contour, n_points))

    if not contours_out:
        # fallback for empty mask
        t = np.linspace(0, 2 * np.pi, n_points)
        r = min(H, W) * 0.1
        contours_out.append(
            np.column_stack([H / 2 + r * np.sin(t), W / 2 + r * np.cos(t)])
        )
    return contours_out


def snap_contour_to_edges(
    snake_init: np.ndarray,
    edge_map: np.ndarray,
    alpha: float = 0.01,
    beta: float = 1.0,
    gamma: float = 0.01,
    max_iter: int = 500,
) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snake = active_contour(
            edge_map,
            snake_init,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            max_num_iter=max_iter,
            boundary_condition="periodic",
        )
    return snake


def contour_to_mask(contour: np.ndarray, shape: tuple) -> np.ndarray:
    from matplotlib.path import Path

    H, W = shape
    path = Path(contour[:, ::-1])
    cols, rows = np.meshgrid(np.arange(W), np.arange(H))
    points = np.column_stack([cols.ravel(), rows.ravel()])
    mask = path.contains_points(points).reshape(H, W)
    return mask.astype(np.uint8)


def morphological_cleanup(mask: np.ndarray, disk_radius: int = 3) -> np.ndarray:
    struct_el = morphology.disk(disk_radius)
    mask = morphology.binary_closing(mask, struct_el)
    mask = morphology.binary_opening(mask, struct_el)
    return mask.astype(np.uint8)


def boundary_edge_score(mask: np.ndarray, edge_map: np.ndarray) -> float:
    ring = morphology.disk(2)
    dilated = morphology.binary_dilation(mask, ring)
    eroded = morphology.binary_erosion(mask, ring)
    boundary = dilated & ~eroded
    if boundary.sum() == 0:
        return 0.0
    return float(edge_map[boundary].mean())


def mask_needs_refinement(
    mask: np.ndarray,
    edge_map: np.ndarray,
    threshold: float = 0.2,
) -> bool:
    score = boundary_edge_score(mask, edge_map)
    return score < threshold


def refine_mask(im: np.ndarray, mask: np.ndarray):
    edge_map = build_edge_map(im)
    blobs = extract_blob_contours(mask)
    combined = np.zeros(im.shape[:2], dtype=np.uint8)

    for snake_init in blobs:
        snake = snap_contour_to_edges(snake_init, edge_map)
        blob_mask = contour_to_mask(snake, im.shape[:2])
        combined = np.maximum(combined, blob_mask)

    combined = morphological_cleanup(combined)
    return combined, edge_map


def save_mask(mask: np.ndarray, path: str) -> None:
    plt.imsave(path, mask, cmap="gray")


def main() -> None:
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_dir = os.path.join(ROOT, "data", "imgs")
    mask_dir = os.path.join(ROOT, "data", "masks")
    output_dir = os.path.join(ROOT, "data", "output_masks")

    os.makedirs(output_dir, exist_ok=True)

    dice_scores: list[float] = []
    n_deleted = 0
    n_skipped = 0
    n_refined = 0

    for filename in sorted(os.listdir(img_dir)):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        image_path = os.path.join(img_dir, filename)
        name, ext = os.path.splitext(filename)
        mask_filename = name + "_mask.png"
        mask_path = os.path.join(mask_dir, mask_filename)

        if not os.path.exists(mask_path):
            os.remove(image_path)
            n_deleted += 1
            print(f"{filename} -> no mask found, image deleted")
            continue

        out_mask_path = os.path.join(output_dir, mask_filename)

        im = load_image(image_path)
        input_mask = load_mask(mask_path)

        edge_map = build_edge_map(im)
        if not mask_needs_refinement(input_mask, edge_map):
            save_mask(input_mask, out_mask_path)
            n_skipped += 1
            print(f"{filename} -> mask already good, copied to output")
            continue

        refined_mask, _ = refine_mask(im, input_mask)
        save_mask(refined_mask, out_mask_path)

        dice = calculate_dice(refined_mask, input_mask)
        dice_scores.append(dice)
        n_refined += 1
        print(f"{filename} -> refined, Dice vs input: {dice:.4f}")

    print("\n--- Summary ---")
    print(f"Deleted (no mask)  : {n_deleted}")
    print(f"Copied as-is       : {n_skipped}")
    print(f"Refined            : {n_refined}")
    if dice_scores:
        print(f"Avg Dice vs input  : {np.mean(dice_scores):.4f}")
        print(f"Std                : {np.std(dice_scores):.4f}")
        print(f"Min / Max          : {np.min(dice_scores):.4f} / {np.max(dice_scores):.4f}")


if __name__ == "__main__":
    main()
