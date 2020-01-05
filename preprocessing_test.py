import numpy as np
from classification.preprocessing import extract_features_rgb_avg
from starter import load_avp_dataset

X, y = load_avp_dataset()

def get_image_of_class(c, i):
    counter = 0
    for j in range(y.shape[0]):
        if y[j] == c:
            counter+=1
            if counter == i:
                return X[j]
    return None


x = get_image_of_class(0,1)

print(extract_features_rgb_avg(x))

