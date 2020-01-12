# -*- coding: utf-8 -*-
import numpy as np
# TODO: You can add additional imports - if you do so, please list all additional packages (including their version) in a separate file called "REQUIREMENTS.txt".
# However, all of your python code must be in this file - you can NOT submit other/more python files than this one!

# ***************************************************************************
# ATTENTION: USE 4 BLANKS FOR ENCODING A TAB - DO NOT CHANGE THIS ENCODING! *
# ***************************************************************************

# *************************************************************************
# CAUTION: DO NOT CHANGE THE NAME OR SIGNATURE OF THE FOLLOWING FUNCTION! *
# *************************************************************************
def my_submission(X_train, y_train, X_test):
    """
    Function containing your model-(pipeline) submission to the "IntroML-2019 competition".

    Parameters
    ----------
    X_train : `numpy.ndarray`
        Training samples, shape `(n_samples_in_X_train, n_features)`.
    y_train : `numpy.ndarray`
        Class labels (`int`) of training samples, shape `(n_samples_in_X_train, )`.
    X_test : `numpy.ndarray`
        Testing samples, shape `(n_samples_in_X_test, n_features)`.

    Returns
    -------
    `numpy.ndarray`
        One dimensional array of predicted class labels (`int`). Must be of size `(n_samples_in_X_test,)`. Make sure that the predicted class labels are integers!
    """

    X_train_features = np.array([extract_features(x) for x in X_train])
    X_test_features = np.array([extract_features(x) for x in X_test])

    nbc = NaiveBayesClassifier()

    nbc.fit(X_train_features, y_train)

    return np.array(pred_all(nbc, X_test_features))

THRESHOLD_BRIGHTNESS = (256.0*3.0)/2.0
def extract_features(x):
    return np.append(extract_feature_highest_rgb(x), extract_feature_brightness_above_threshold(x, THRESHOLD_BRIGHTNESS))

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


def pred_all(classifier, X_pred):
    y_pred = []
    for x in X_pred:
        y_pred.append(classifier.predict(x))
    return y_pred

class NaiveBayesClassifier:

    def __init__(self):
        pass

    def fit(self,X,Y):
        #Pc is a dictionary mapping c to the probability P(Y=c)
        classes, occurence = np.unique(Y,return_counts=True)
        self.classes = classes
        probability = []
        #print("Occurence: " + str(occurence))
        for i in range(occurence.shape[0]):
            probability.append(float(occurence[i])/Y.size)
        self.Pc = dict(zip(classes, probability))
        #print(self.Pc)
        self.xAvg = {}
        for c in classes:
            self.xAvg[c] = np.divide(np.sum(np.array([X[i] for i in np.where(Y==c)[0]]), axis=0),np.array([X.shape[1]]*X.shape[1]))

    def predict(self,x):
        probability = {}

        maxClass = None
        maxProb = 0
        for c in self.classes:
            probability[c]=(1 - np.divide(np.sum(np.abs(np.subtract(x,self.xAvg[c]))), x.shape[0])) #* self.Pc[c]
            print(str(c) + " :" + str(probability[c]))
            if probability[c] > maxProb:
               maxProb = probability[c]
               maxClass = c

        return maxClass
