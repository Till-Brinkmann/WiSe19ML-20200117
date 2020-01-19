import numpy as np

def extract_feature_rgb_avg(x):
    sum_rgb = [0,0,0]
    for col in x:
        for rgb in col:
            sum_rgb[0] += rgb[0]
            sum_rgb[1] += rgb[1]
            sum_rgb[2] += rgb[2]
    return np.divide(np.array(sum_rgb),x.shape[0]*x.shape[1])

def extract_feature_local_difference(x, threshold):
    feat = np.zeros(x.shape[0]*x.shape[1])

    maxI = x.shape[0]-1
    maxJ = x.shape[1]-1

    print("Extraction")
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            diff = np.zeros(x.shape[2])
            if i != 0:
                diff += np.abs(x[i][j] - x[i-1][j])
            if i != maxI:
                diff += np.abs(x[i][j] - x[i+1][j])
            if j != 0:
                diff += np.abs(x[i][j] - x[i][j-1])
            if j != maxJ:
                diff += np.abs(x[i][j] - x[i][j+1])
            if np.sum(diff) > threshold:
                feat[i * maxI + j] = 1
    return feat

def extract_feature_highest_rgb(x):
    feat = np.zeros(x.shape[0]*x.shape[1]*3, dtype=np.uint8)
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
            i += 3
    return feat

def extract_feature_brightness_above_threshold(x, threshold):
    feat = np.zeros(x.shape[0]*x.shape[1], dtype=np.uint8)

    i = 0
    for col in x:
        for rgb in col:
            if int(rgb[0])+int(rgb[1])+int(rgb[2]) > threshold:
                feat[i] = 1
            i += 1
    return np.array(feat)
