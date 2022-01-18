import numpy as np
import sklearn.datasets

bc = sklearn.datasets.load_breast_cancer() #loading datasets

X = bc.data
Y = bc.target
#print(X)
#print(Y)

#print(X.shape,Y.shape)

import pandas as pd
data = pd.DataFrame(bc.data,columns = bc.feature_names)
data['class'] = bc.target
print(data.head) #see the data set class wise and features
print(data.describe()) #statistical data

print(data['class'].value_counts())
print(bc.target_names)
print(data.groupby('class').mean())

#model selection
from sklearn.model_selection import train_test_split

# 10% test size
# stratify - for correct distribution of data as of the original data
# random state - specific split of data. each value of random state spplits data differently
X_train , X_test , Y_train ,Y_test = train_test_split(X,Y,test_size=0.1
,stratify=Y,random_state=1)
print(Y.mean(),Y_train.mean(),Y_test.mean()) # stratify
print(X.mean(),X_train.mean(),X_test.mean())
print(X_train)

#import logistic regression from sklearn
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression() #load logisticRegression into variable classifier
classifier.fit(X_train,Y_train)# training the model

#import accuracy score
from sklearn.metrics import accuracy_score
pred_train_data = classifier.predict(X_train)
accuracy_train_data = accuracy_score(Y_train,pred_train_data)
print("Accuracy of training data is: ",accuracy_train_data)

#prediction on test data
pred_test_data = classifier.predict(X_test)
accuracy_test_data = accuracy_score(Y_test,pred_test_data)
print("Accuracy of testing data is: ",accuracy_test_data)

input = (17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,
0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,
184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189) #for data kindly download the dataset from kaggle.
inputData = np.asarray(input)
#print(inputData)

inputReshaped = inputData.reshape(1,-1)
print(inputReshaped)

#prediction 
prediction = classifier.predict(inputReshaped)
print(prediction)
if prediction[0]==0:
    print("This is malignent")
else:
    print("This is benign")