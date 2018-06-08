# Weather-Prediction

This project is basically we will predict weather there is rainfall at a location using a data set based on (processed) infrared satellite image information. There are 200,000 observation / sample points with 14 features.
Training data: 126,551 samples of class 0 (no rain) and 73,449 samples of class 1 (rain)

Learners we used in this project
a.	Logistic Regression
b.	K-Nearest Neighbors
c.	Random Forest
d.	Neural Network

We divided the data set in X_train.txt into two parts - training data set and validation data set- with ratio 80% and 20% respectively so that we can train our data and test the model on validation data set before actually testing on the real test data set, X_test.txt at Kaggle.
We don’t have test data of Y, so we used error rate(misclassification) of training, validation, Kaggle test, and our goal is to get highest accuracy on Kaggle test data set.
