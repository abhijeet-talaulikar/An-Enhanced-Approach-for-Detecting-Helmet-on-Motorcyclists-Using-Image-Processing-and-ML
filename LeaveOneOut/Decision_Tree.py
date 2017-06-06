import numpy as np
from sklearn.model_selection import LeaveOneOut
import pydotplus
from sklearn import tree
from sklearn.metrics import *
from timeit import default_timer as timer
from sklearn.feature_selection import *

helmet_data = np.genfromtxt ('helmet.csv', delimiter=",")
face_data = np.genfromtxt ('face.csv', delimiter=",")

data = np.concatenate((helmet_data, face_data), 0)
np.random.shuffle(data) #shuffle the tuples

#feature reduction (on HOG part)
gain, j = mutual_info_classif(data[:, 8:-1], data[:, -1], discrete_features='auto', n_neighbors=3, copy=True, random_state=None), 0
for i in np.arange(len(gain)):
	if gain[i] <= 0.001:
		data = np.delete(data, 8+i-j, 1)
		j += 1

precision, recall, f1, accuracy, support, fn = 0, 0, 0, 0, 0, 0

loo = LeaveOneOut()

start = timer()
for train, test in loo.split(data):
	X_train, X_test = data[train, 0:-1], data[test, 0:-1]
	y_train, y_test = data[train, -1], data[test, -1]
	clf = tree.DecisionTreeClassifier().fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	precision += precision_score(y_test, y_pred, average = 'micro')
	recall += recall_score(y_test, y_pred, average = 'micro')
	f1 += f1_score(y_test, y_pred, average = 'micro')
	accuracy += accuracy_score(y_test, y_pred)
	y = y_test - y_pred
	fn += sum(y[y > 0]) / len(y_test)
end = timer()

precision /= data.shape[0]
recall /= data.shape[0]
f1 /= data.shape[0]
accuracy /= data.shape[0]
fn /= data.shape[0]

print("Precision \t: %s" % round(precision, 4))
print("Recall \t\t: %s" % round(recall, 4))
print("F1 \t\t: %s" % round(f1, 4))
print("Accuracy \t: %s" % round(accuracy, 4))
print("False Neg \t: %s%%" % round(fn * 100, 4))
print("\nExecution time: %s ms" % round((end - start) * 1000, 4))
