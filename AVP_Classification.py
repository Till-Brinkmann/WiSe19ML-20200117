import numpy as np
from classification.classifier import NaiveBayesClassifier
from classification.preprocessing import *
from sklearn.metrics import classification_report, f1_score
from starter import load_avp_dataset

def pred_all(classifier, X_pred):
    y_pred = []
    for x in X_pred:
        y_pred.append(classifier.predict(x))
    return y_pred


THRESHOLD_BRIGHTNESS = (256*3.0)/2.5
THRESHOLD_DIFFERENCE = 1000

def extract_features(x):
    return np.append(extract_feature_highest_rgb(x), extract_feature_brightness_above_threshold(x, THRESHOLD_BRIGHTNESS))
    #return extract_feature_local_difference(x, THRESHOLD_DIFFERENCE)
    #return np.append(extract_feature_local_difference(x, THRESHOLD_DIFFERENCE), extract_feature_brightness_above_threshold(x, THRESHOLD_BRIGHTNESS))
    #return extract_feature_brightness_above_threshold(x, THRESHOLD_BRIGHTNESS)
    #return extract_feature_highest_rgb(x)



X, y = load_avp_dataset()

shuffled = np.array([(extract_features(X[i]),y[i]) for i in range(X.shape[0])])
np.random.shuffle(shuffled)
X = np.array([shuffled[i][0] for i in range(X.shape[0])])
y = np.array([shuffled[i][1] for i in range(y.shape[0])])

testSize = 50

X_split = np.split(X,[X.shape[0]-testSize])
y_split = np.split(y,[y.shape[0]-testSize])

X_train = X_split[0]
X_test = X_split[1]

y_train = y_split[0]
y_test = y_split[1]

nbc = NaiveBayesClassifier()

print("Starting Training")
nbc.fit(X_train, y_train)
print("Training finished")

y_pred = pred_all(nbc, X_test)

print("Test: " + str(y_test))
print("Pred: " + str(y_pred))

print(classification_report(y_test,y_pred))

print("F1-score (macro): " + str(f1_score(y_test, y_pred, average="macro")))
