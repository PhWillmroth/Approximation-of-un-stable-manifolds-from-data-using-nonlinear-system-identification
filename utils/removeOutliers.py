import pandas as pd
import numpy as np

# 'm' is the threshold; all samples are ignored whose distance to the median is larger than (m * median)
def removeOutliers(data, m=2.):
    if isinstance(data, pd.DataFrame):
        data = data.values
    elif isinstance(data, np.ndarray):
        pass
    else:
        raise TypeError(f'The function "removeOutliers" only takes numpyArrays or pandasDataFrames. You provided "{data.__class__}".')

    data.sort() # sorting columnwise so the numbers are increasing
    
    # remove outliers
    d = np.abs(data - np.median(data))
    mdev = np.median(d) # The median OF the distance to the median
    s = d / (mdev if mdev else 1.) # distance to median / (median of the distance to the median) = relative distance to median
    return data[s < m]