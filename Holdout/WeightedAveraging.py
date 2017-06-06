import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
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

clf1 = svm.SVC(kernel='linear', probability=True).fit(X_train, y_train)
y_prob1 = clf1.predict_proba(X_test)[:,1]
y_pred1 = clf1.predict(X_test)
y_acc1 = accuracy_score(y_test, y_pred1)

clf2 = MLPClassifier(solver='lbfgs', activation = 'logistic', learning_rate = 'adaptive', hidden_layer_sizes = (5), random_state = 1)
clf2 = clf2.fit(X_train, y_train)
y_prob2 = clf2.predict_proba(X_test)[:,1]
y_pred2 = clf2.predict(X_test)
y_acc2 = accuracy_score(y_test, y_pred2)

clf3 = LogisticRegression().fit(X_train, y_train)
y_prob3 = clf3.predict_proba(X_test)[:,1]
y_pred3 = clf3.predict(X_test)
y_acc3 = accuracy_score(y_test, y_pred3)

y_prob, y_pred = np.zeros(len(y_test)), np.zeros(len(y_test))
for i in np.arange(len(y_test)):
	y_prob[i] = ((y_acc1*y_prob1[i])+(y_acc2*y_prob2[i])+(y_acc3*y_prob3[i])) / (y_acc1+y_acc2+y_acc3)

for i in np.arange(len(y_test)):
	y_pred[i] = (y_prob[i] > 0.5)

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
fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=1)
roc_auc = auc(fpr, tpr)
plt.title('Weighted Averaging')
plt.plot(fpr, tpr, 'b', label='AUC = %s'% round(roc_auc, 4))
print("\nAUC \t: %s" % round(roc_auc, 4))
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.05,1.0])
plt.ylim([0.0,1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
