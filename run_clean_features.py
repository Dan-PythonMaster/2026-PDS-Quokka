import pandas as pd
import skimage as ski
import numpy as np
import os
from src.feature_A import compute_asymmetry
from src.feature_B import border_calc
from src.feature_C import extract_color_features

ROOT     = os.path.dirname(os.path.abspath(__file__))
img_dir  = os.path.join(ROOT, "data", "imgs_clean")
mask_dir = os.path.join(ROOT, "data", "masks")

# The CSV just gotta have a img_id and diagnostic column
PATH = os.path.join(ROOT, "data", "metadata.csv")
df   = pd.read_csv(PATH)

# Match metadata against available cleaned images
clean_files = [f for f in os.listdir(img_dir) if f.endswith(".png")]
clean_map   = {f.replace("clean_", ""): f for f in clean_files}

df    = df[df["img_id"].isin(clean_map)][["img_id", "diagnostic"]].reset_index(drop=True)
imgID = df["img_id"].to_list()

asymmetry_features = []
border_features    = []
color_features     = []



for i in range(len(imgID)):
    img_path  = os.path.join(img_dir, clean_map[imgID[i]])
    name, ext = os.path.splitext(imgID[i])
    mask_path = os.path.join(mask_dir, f"{name}_mask{ext}")

    if not os.path.exists(mask_path):
        continue

    img = ski.io.imread(img_path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    img = ski.transform.resize(img, (255, 255))

    mask = ski.io.imread(mask_path)
    mask = ski.transform.resize(mask, (255, 255), preserve_range=True)

    # mask to boolean
    if mask.ndim == 3:
        mask = ski.color.rgb2gray(mask) > 0.5
    else:
        mask = mask > 0.5

    # asymmetry features
    asymmetry_features.append(compute_asymmetry(mask))

    # border features
    border_features.append(border_calc(mask))

    # colour features
    color_features.append(extract_color_features(img, mask))

# Add to dataframe
asymmetry_df = pd.DataFrame(asymmetry_features)
border_df    = pd.DataFrame(border_features)
color_df     = pd.DataFrame(color_features)
df           = pd.concat([df, asymmetry_df, border_df, color_df], axis=1)

# Save as feature extracted CSV
PATH = os.path.join(ROOT, "data", "features_clean.csv")
df.to_csv(PATH)