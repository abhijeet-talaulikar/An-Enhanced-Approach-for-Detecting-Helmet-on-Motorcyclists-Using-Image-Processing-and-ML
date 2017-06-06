import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
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
	
	clf1 = svm.SVC(kernel='linear', probability=True).fit(X_train, y_train)
	y_prob1 = clf1.predict_proba(X_test)[:,1]
	y_pred1 = clf1.predict(X_test)
	y_acc1 = accuracy_score(y_test, y_pred1)

	clf2 = KNeighborsClassifier(n_neighbors = 3).fit(X_train, y_train)
	y_prob2 = clf2.predict_proba(X_test)[:,1]
	y_pred2 = clf2.predict(X_test)
	y_acc2 = accuracy_score(y_test, y_pred2)

	clf3 = LogisticRegression().fit(X_train, y_train)
	y_prob3 = clf3.predict_proba(X_test)[:,1]
	y_pred3 = clf3.predict(X_test)
	y_acc3 = accuracy_score(y_test, y_pred3)

	clf4 = GaussianNB().fit(X_train, y_train)
	y_prob4 = clf4.predict_proba(X_test)[:,1]
	y_pred4 = clf4.predict(X_test)
	y_acc4 = accuracy_score(y_test, y_pred4)

	y_prob, y_pred = np.zeros(len(y_test)), np.zeros(len(y_test))
	for i in np.arange(len(y_test)):
		y_prob[i] = ((y_acc1*y_prob1[i])+(y_acc2*y_prob2[i])+(y_acc3*y_prob3[i])+(y_acc4*y_prob4[i])) / (y_acc1+y_acc2+y_acc3+y_acc4)

	for i in np.arange(len(y_test)):
		y_pred[i] = (y_prob[i] > 0.5)
	
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
