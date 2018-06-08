import numpy as np
import mltools as ml
import matplotlib.pyplot as plt

X = np.genfromtxt("data/X_train.txt", delimiter=None)
Y = np.genfromtxt("data/Y_train.txt", delimiter=None)
Xt,Xv,Yt,Yv = ml.splitData(X,Y,0.01)

err_k_t = [None]*15
err_k_v = [None]*15

for i in range (1, 15, 1):
    Xt, Xv, Yt, Yv = ml.splitData(X[:,0:i], Y, 0.01)
    knn = ml.knn.knnClassify()
    knn.train(Xt, Yt)
    knn.K = 3
    print i
    err_k_t[i] = knn.err(Xt,Yt)
    err_k_v[i] = knn.err(Xv,Yv)
print err_k_t, err_k_v
plt.plot(err_k_t, 'g-', err_k_v, 'r')
plt.legend(('Training Error Rate', 'Validation Error Rate'),'upper right')
#plt.xlabel('number of neighbor K')
plt.xlabel('number of feature')
plt.ylabel('error rate')
plt.show()

"""knn = ml.knn.knnClassify()
knn.train(Xt, Yt)
knn.K = 3
Xte = np.genfromtxt("data/X_test.txt", delimiter=None)
Ypred = knn.predictSoft(Xte)
np.savetxt('Yhat_KNN.txt', np.vstack( (np.arange(len(Ypred)) , Ypred[:,1]) ).T, '%d, %.2f',header='ID,Prob1',comments='',delimiter=',');"""
