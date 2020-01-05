import numpy as np

data = np.load("avp_dataset.npz")

X = data["X"]
y = data["y"]

classes, occurence = np.unique(y,return_counts=True)
data_classes = dict(zip(classes,occurence))

print("dataset")
print("class distribution:" + str(data_classes))
print("")

print("Image Dimension: " + str(X[0].shape))
