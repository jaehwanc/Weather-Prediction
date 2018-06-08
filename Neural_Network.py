import numpy as np
import mltools as ml
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
X = np.genfromtxt("data/X_train.txt", delimiter=None)
Y = np.genfromtxt("data/Y_train.txt", delimiter=None)

Xtr,Xv,Ytr,Yv = ml.splitData(X,Y,0.8)
err_t = [None]*10
err_v = [None]*10
for i in range(1,10,1):
    Xtr, Xv, Ytr, Yv = ml.splitData(X[:,0:i], Y, 0.8)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=[14]*i, random_state=1)
    clf.fit(Xtr, Ytr)
    Yet = clf.predict(Xtr)
    err_t[i] = np.mean(Yet!=Ytr)
    Yev = clf.predict(Xv)
    err_v[i] = np.mean(Yev!=Yv)
print err_t ,err_v
plt.plot(err_t, 'g-', err_v, 'r-')
plt.legend(('Training Error Rate', 'Validation Error Rate'),'upper left')
plt.xlabel('Hidden Layer Size')
#plt.xlabel('Number of Features')
plt.ylabel('error rate')
plt.show()

"""Xte = np.genfromtxt("data/X_test.txt", delimiter=None)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(14,14,14,14,14), random_state=1)
clf.fit(Xtr, Ytr)
Ypred = clf.predict(Xte)
np.savetxt('Yhat_Neural_Network.txt',
np.vstack( (np.arange(len(Ypred)) , Ypred) ).T, '%d, %.2f',header='ID,Prob1',comments='',delimiter=',');"""
