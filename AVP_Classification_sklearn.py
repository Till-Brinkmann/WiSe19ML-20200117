import numpy as np
from classification.preprocessing import *
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report
from starter import load_avp_dataset

def pred_all(classifier, X_pred):
    y_pred = []
    for x in X_pred:
        y_pred.append(classifier.predict(x))
    return y_pred

THRESHOLD_BRIGHTNESS = (256*3.0)/2.0

def extract_features(x):
    #return extract_feature_brightness_above_threshold(x, THRESHOLD_BRIGHTNESS)
    return extract_feature_highest_rgb(x)

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

nbc = BernoulliNB()

print("Starting Training")
nbc.fit(X_train, y_train)
print("Training finished")

y_pred = nbc.predict(X_test)#pred_all(nbc, X_test)

print("Test: " + str(y_test))
print("Pred: " + str(y_pred))

print(classification_report(y_test,y_pred))
