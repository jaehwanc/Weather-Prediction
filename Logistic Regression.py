import numpy as np
import mltools as ml
import matplotlib.pyplot as plt
import random
import mltools.logistic2 as lc2
reload(lc2)

X = np.genfromtxt("data/X_train.txt",delimiter=None)
Y = np.genfromtxt("data/Y_train.txt",delimiter=None)
learner = lc2.logisticClassify2();


Xt,Xv,Yt,Yv = ml.splitData(X[:,0:14],Y,0.8)
Xt,Yt = ml.shuffleData(Xt, Yt)
Xt,_ = ml.transforms.rescale(Xt)
learner.classes = np.unique(Yt)
wts = [0.5 ,1 ,-0.25,((random.random()-0.5)*2),((random.random()-0.5)*2),((random.random()-0.5)*2),((random.random()-0.5)*2),((random.random()-0.5)*2),((random.random()-0.5)*2),((random.random()-0.5)*2),((random.random()-0.5)*2),((random.random()-0.5)*2),((random.random()-0.5)*2),((random.random()-0.5)*2),((random.random()-0.5)*2)];
#wts = np.append(wts,[((random.random()-0.5)*2)])
#wts = [0.5 ,1]
learner.theta = wts
lc2.train(learner, Xt, Yt, 0.01, 1e-5, 10000, plot=1, reg=0)
plt.show()
print learner.err(Xt, Yt)
Xte = np.genfromtxt("data/X_test.txt", delimiter=None)
Yt =  np.genfromtxt("data/Y_train.txt", delimiter=None)
Ypred = lc2.train(learner, Xte, Yt, 0.01, 1e-5, 10000, plot=1, reg=0)
np.savetxt('Yhat_Logistic_Regression.txt',
np.vstack( (np.arange(len(Ypred)) , Ypred) ).T, '%d, %.2f',header='ID,Prob1',comments='',delimiter=',');

"""X=[1,3,6,9,12,14]
Y1=[0.340225,0.3366375, 0.33399375, 0.32776875, 0.32600755, 0.32456875]
Y2=[0.36625, 0.36625, 0.369475, 0.364475, 0.365575, 0.36625]
plt.plot(X, Y1, 'g-', X, Y2, 'r-')
plt.legend(('Training Error Rate', 'Validation Error Rate'),'center right')
plt.xlabel('number of feature')
plt.ylabel('error rate')
plt.show()"""

"""from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.coef_ = [1,-0.5,0.25,1,-0.5,0.25,1,-0.5,0.25,1,-0.5,0.25,1,-0.5]
Ypred = lr.predict_proba(Xt)
print Ypred.shape, Ypred"""



