import numpy as np
import mltools as ml
import matplotlib.pyplot as plt

X = np.genfromtxt("data/X_train.txt", delimiter=None)
Y = np.genfromtxt("data/Y_train.txt", delimiter=None)
Xt,Xv,Yt,Yv = ml.splitData(X,Y,0.8)

err_depth_tr = [None]*15
err_depth_v = [None]*15
for d in range(1,15,1):
    dt = ml.dtree.treeClassify(Xt,Yt,minLeaf=8, minParent = 16, maxDepth = d)
    err_depth_tr[d] = ml.dtree.treeClassify.err(dt,Xt,Yt)
    err_depth_v[d] = ml.dtree.treeClassify.err(dt,Xv,Yv)
xs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
plt.plot(xs, err_depth_tr, '-r', xs, err_depth_v, '-b')
plt.legend(('Training Error Rate', 'Validation Error Rate'),'upper right')
plt.xlabel('Maximum Depth')
plt.ylabel('error rate')
plt.show()

rf = [None]*25
Yt_hat = np.zeros((Yt.shape[0],25))
Yv_hat = np.zeros((Yv.shape[0],25))

for i in range(0,24,1):
    [Xi, Yi] = ml.bootstrapData(Xt, Yt, Xt.shape[0])
    rf[i] = ml.dtree.treeClassify(Xi, Yi, minLeaf=8, minParent = 512, maxDepth = 7, nFeatures = 14)
    Yt_hat[:, i] = rf[i].predict(Xt)
    Yv_hat[:, i] = rf[i].predict(Xv)

err_e_t = [None]*6
err_e_v = [None]*6

Yt_hat_e = Yt_hat[:, 0]
err_e_t[0] = np.mean(Yt_hat_e.reshape(Yt.shape) != Yt)
Yv_hat_e = Yv_hat[:, 0]
err_e_v[0] = np.mean(Yv_hat_e.reshape(Yv.shape) != Yv)

j=1
for i in [5, 10, 15, 20, 25]:
    Yt_hat_e = (np.mean(Yt_hat[:,0:i], axis=1)>0.5)
    Yt_hat_e = Yt_hat_e.astype(int)
    err_e_t[j] = np.mean(Yt_hat_e.reshape(Yt.shape) != Yt)
    Yv_hat_e = (np.mean(Yv_hat[:,0:i], axis=1)>0.5)
    Yv_hat_e = Yv_hat_e.astype(int)
    err_e_v[j] = np.mean(Yv_hat_e.reshape(Yv.shape) != Yv)
    j=j+1
print err_e_t, err_e_v

xs = [1,5,10,15,20,25]
plt.plot(xs, err_e_t, '-r', xs, err_e_v, '-b')
plt.legend(('Training Error Rate', 'Validation Error Rate'),'upper right')
plt.xlabel('number of ensembles')
plt.ylabel('error rate')
plt.show()

rf = [None]*25
Xte = np.genfromtxt("data/X_test.txt", delimiter=None)
Yte = np.zeros((Xte.shape[0],25))
for i in range(0,24,1):
    [Xi, Yi] = ml.bootstrapData(Xv, Yv, Xv.shape[0])
    rf[i] = ml.dtree.treeClassify(Xi, Yi, minLeaf=8, minParent=512, maxDepth=7, nFeatures=14)
    Yte[:,i] = rf[i].predictSoft(Xte)[:,1]
Yte = np.array(np.mean(Yte, axis = 1))
# Now output a file with two columns, a row ID and a confidence in class 1:
np.savetxt('Yhat_random_forest2.txt',
np.vstack( (np.arange(len(Yte)) , Yte) ).T, '%d, %.2f',header='ID,Prob1',comments='',delimiter=',');
