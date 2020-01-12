import numpy as np
from submission import my_submission
from sklearn.metrics import classification_report

def load_avp_dataset(file_path="avp_dataset.npz"):
    data = np.load(file_path)
    return data["X"], data["y"]
X, y = load_avp_dataset()

#shuffled = np.array([(extract_features(X[i]),y[i]) for i in range(X.shape[0])])
#np.random.shuffle(shuffled)
#X = np.array([shuffled[i][0] for i in range(X.shape[0])])
#y = np.array([shuffled[i][1] for i in range(y.shape[0])])

testSize = 50

X_split = np.split(X,[X.shape[0]-testSize])
y_split = np.split(y,[y.shape[0]-testSize])

X_train = X_split[0]
X_test = X_split[1]

y_train = y_split[0]
y_test = y_split[1]

y_pred = my_submission(X_train, y_train, X_test)

print(classification_report(y_test,y_pred))
