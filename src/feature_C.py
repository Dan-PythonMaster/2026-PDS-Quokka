import skimage as ski
import numpy as np

def compactness_calc(mask):
    #compactness
    area = mask.sum()
    if area == 0:
        return np.nan
    perimeter = area - ski.morphology.erosion(mask,ski.morphology.disk(3)).sum()  # erosion can be refined to be even finer
    compactness = perimeter**2 / (area * 12)
    return compactness