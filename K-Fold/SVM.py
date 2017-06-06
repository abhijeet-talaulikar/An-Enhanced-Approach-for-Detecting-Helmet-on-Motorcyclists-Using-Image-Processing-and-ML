import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import *
from timeit import default_timer as timer
from random import randint
from sklearn.feature_selection import *
from sklearn.decomposition import PCA

helmet_data = np.genfromtxt ('helmet.csv', delimiter=",")
face_data = np.genfromtxt ('face.csv', delimiter=",")

data_full = np.concatenate((helmet_data, face_data), 0)
np.random.shuffle(data_full) #shuffle the tuples

#feature reduction (on HOG part)
#gain, j = mutual_info_classif(data_full[:, 8:-1], data_full[:, -1], discrete_features='auto', n_neighbors=3, copy=True, random_state=None), 0
#for i in np.arange(len(gain)):
#	if gain[i] <= 0.001:
#		data_full = np.delete(data_full, 8+i-j, 1)
#		j += 1
#data = np.copy(data_full)

#principal component analysis
pca = PCA(n_components=100)
data = pca.fit_transform(data_full[:, 8:-1])
data = np.concatenate((data_full[:, 0:8], data, np.array([data_full[:, -1]]).T), axis=1)

precision, recall, f1, accuracy, support, fn, roc_auc = 0, 0, 0, 0, 0, 0, 0
colors = ['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange']

k = 10
kf = KFold(n_splits = k)

start = timer()
for train, test in kf.split(data):
	X_train, X_test = data[train, 0:-1], data[test, 0:-1]
	y_train, y_test = data[train, -1], data[test, -1]
	clf = svm.SVC(kernel='linear', probability=True).fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	
	#ROC curve
	y_prob = clf.predict_proba(X_test)[:,1]
	fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=1)
	roc_auc += auc(fpr, tpr)
	plt.plot(fpr, tpr, color=colors[randint(0, len(colors)-1)])
	
	precision += precision_score(y_test, y_pred, average = 'macro')
	recall += recall_score(y_test, y_pred, average = 'macro')
	f1 += f1_score(y_test, y_pred, average = 'macro')
	accuracy += accuracy_score(y_test, y_pred)
	y = y_test - y_pred
	fn += sum(y[y > 0]) / len(y_test)
end = timer()

precision /= k
recall /= k
f1 /= k
accuracy /= k
fn /= k

print("Precision \t: %s" % round(precision, 4))
print("Recall \t\t: %s" % round(recall, 4))
print("F1 \t\t: %s" % round(f1, 4))
print("Accuracy \t: %s" % round(accuracy, 4))
print("False Neg \t: %s%%" % round(fn * 100, 4))
print("Mean AUC \t: %s" % round(roc_auc / k, 4))
print("\nExecution time: %s ms" % round((end - start) * 1000, 4))

#ROC curve
plt.title('Support Vector Machines (AUC = %s)' % round(roc_auc, 4))
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.05,1.0])
plt.ylim([0.0,1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
