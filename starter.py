#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def plot_image(img):
    plt.figure()
    plt.imshow(img)
    plt.show()


def load_avp_dataset(file_path="avp_dataset.npz"):
    data = np.load(file_path)
    return data["X"], data["y"]


if __name__ == "__main__":
    # Load data
    X, y = load_avp_dataset()

    # Plot picture 42
    i = int(input())   # Counting starts at 0!
    if i < 0:
        counter = 0
        for j in range(y.shape[0]):
            if y[j] == abs(i+1):
                counter+=1
                if counter <= 10:
                    plot_image(X[j])
    else:
        img = X[i]
        out = y[i]
        print(str(img))
        print(str(out))
        plot_image(img)
