import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import *
from timeit import default_timer as timer
from sklearn.metrics import roc_curve, auc
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

X_train, X_test, y_train, y_test = train_test_split(data[:, 0:-1], data[:, -1], test_size = 0.4, random_state = 0)

start = timer()
clf = svm.SVC(kernel='linear', probability=True).fit(X_train, y_train)
y_pred = clf.predict(X_test)
end = timer()

print("Confusion Matrix: \n")
print(confusion_matrix(y_test, y_pred))

target_names = ['Helmet', 'No Helmet']
print("\n\nClassification Report: \n")
print("Accuracy: %s" % round(accuracy_score(y_test, y_pred), 4))
print("Precision \t: %s" % round(precision_score(y_test, y_pred, average = 'macro'), 4))
print("Recall \t\t: %s" % round(recall_score(y_test, y_pred, average = 'macro'), 4))
print("F1 \t\t: %s" % round(f1_score(y_test, y_pred, average = 'macro'), 4))

#Percentage of False Negatives
y = y_test - y_pred
fn = sum(y[y > 0]) * 100 / len(y_test)
print("There are %s%% False Negatives" % round(fn, 4))

print("\nExecution time: %s ms" % round((end - start) * 1000, 4))

#ROC curve
y_prob = clf.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=1)
roc_auc = auc(fpr, tpr)
plt.title('Support Vector Machines')
plt.plot(fpr, tpr, 'b', label='AUC = %s'% round(roc_auc, 4))
print("\nAUC \t: %s" % round(roc_auc, 4))
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.05,1.0])
plt.ylim([0.0,1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
