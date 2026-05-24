from skimage import color 
import numpy as np

def extract_color_features(img_rgb, mask):
    """
    Extract colour features from a skin lesion image.
    
    Parameters:
        img_rgb : np.ndarray, shape (H, W, 3), dtype float, values 0.0-1.0
        mask    : np.ndarray, shape (H, W), dtype bool, True = lesion pixel
    
    Returns:
        dict of feature_name -> float value
    """
    lesion_pixels = img_rgb[mask]      
    surrounding   = img_rgb[~mask]     

    FEATURE_NAMES = [
    'rgb_R_mean', 'rgb_G_mean', 'rgb_B_mean',
    'rgb_R_std',  'rgb_G_std',  'rgb_B_std',
    'hsv_H_mean', 'hsv_S_mean', 'hsv_V_mean',
    'lab_L_mean', 'lab_A_mean', 'lab_B_mean',
    'color_nonuniformity',
    'contrast_diff',
    'contrast_ratio',
    'contrast_standardized',
    'skin_tone_proxy'
    ]

    if len(lesion_pixels) == 0:
       
        return {k: np.nan for k in FEATURE_NAMES}

    features = {}


    for i, ch in enumerate(['R', 'G', 'B']):
        features[f'rgb_{ch}_mean'] = lesion_pixels[:, i].mean()
        features[f'rgb_{ch}_std']  = lesion_pixels[:, i].std()
      
    img_hsv    = color.rgb2hsv(img_rgb)
    lesion_hsv = img_hsv[mask]
    features['hsv_H_mean'] = lesion_hsv[:, 0].mean()  
    features['hsv_S_mean'] = lesion_hsv[:, 1].mean()  
    features['hsv_V_mean'] = lesion_hsv[:, 2].mean() 

    img_lab = color.rgb2lab(img_rgb)
    lesion_lab = img_lab[mask]
    features['lab_L_mean'] = lesion_lab[:, 0].mean()  
    features['lab_A_mean'] = lesion_lab[:, 1].mean()  
    features['lab_B_mean'] = lesion_lab[:, 2].mean()  
   
    features['color_nonuniformity'] = lesion_pixels.std(axis=0).mean()
    # Open Question
    if len(surrounding) > 100:
        les_mean  = lesion_pixels.mean()
        surr_mean = surrounding.mean()
        surr_std  = surrounding.std()

        # Method 1: Subtraction — how much brighter/darker is the lesion
        features['contrast_diff'] = les_mean - surr_mean

        # Method 2: Ratio — less sensitive to overall image brightness
        features['contrast_ratio'] = les_mean / (surr_mean + 1e-6)

        # Method 3: Standardized — z-score style, accounts for skin variation
        features['contrast_standardized'] = (les_mean - surr_mean) / (surr_std + 1e-6)

        # Skin tone proxy: mean L* of pixels outside lesion
        # Higher = lighter surrounding skin, used to group images into tertiles
        features['skin_tone_proxy'] = color.rgb2lab(img_rgb)[~mask][:, 0].mean()

    else:
        features['contrast_diff']         = np.nan
        features['contrast_ratio']        = np.nan
        features['contrast_standardized'] = np.nan
        features['skin_tone_proxy']       = np.nan

    return features
