import numpy as np

def extract_feature_rgb_avg(x):
    sum_rgb = [0,0,0]
    for col in x:
        for rgb in col:
            sum_rgb[0] += rgb[0]
            sum_rgb[1] += rgb[1]
            sum_rgb[2] += rgb[2]
    return np.divide(np.array(sum_rgb),x.shape[0]*x.shape[1])


def extract_feature_highest_rgb(x):
    feat = np.zeros(np.prod(x.shape))
    i = 0
    for col in x:
        for rgb in col:
            if rgb[0] > rgb[1]:
                if rgb[0] > rgb[2]:
                    feat[i] = 1
                else:
                    feat[i+2] = 1
            else:
                if rgb[1] > rgb[2]:
                    feat[i+1] = 1
                else:
                    feat[i+2] = 1
    return feat

def extract_feature_brightness_above_threshold(x, threshold):
    feat = []
    for col in x:
        for rgb in col:
            if int(rgb[0])+int(rgb[1])+int(rgb[2]) > threshold:
                feat.append(1)
            else:
                feat.append(0)
    return np.array(feat)

