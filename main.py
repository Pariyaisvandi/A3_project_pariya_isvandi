#project 3 
#pariya_isvandi

from sklearn.datasets import load_breast_cancer
data=load_breast_cancer()
x=data.data
y=data.target

data.feature_names

data.target_names



#----------EXAMPLE-------
from sklearn.model_selection import KFold
kf= KFold(n_splits=5,shuffle=True,random_state=42)

#-----MODEL SELECTION
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
my_params= { 'n_neighbors':[3,5,7,8,9],
            'metric':['minkowski'  , 'euclidean' , 'manhattan'],
           'weights':['uniform','distance']}

from sklearn.model_selection import GridSearchCV

gs=GridSearchCV(model,my_params,cv=kf,scoring='accuracy')
gs.fit(x,y)
print(gs.best_score_)
print(gs.best_params_)



#---------------------
















'''
from sklearn.metrics import accuracy_score
train_score=accuracy_score(y_true, y_pred)
print(train_score) 

#test_score

y_pred=model.predict(x_test)
test_score=accuracy_score(y_pred)
print(test_score) 

y_true(y_test)

from sklearn.metrics import confusion_matrics
y_pred=model.predict(x_test) 
score=confusion_matrix(y_test,y_pred) 
print(score) 

a=model.coef_
b=model.intercept_
print(a) 
print(b) 
'''
#KNN 
#step0 ---- cleaning
#step1 ----
import numpy as np
#x=np.array(data('load_breast_cancer'))
#y=np.array(data('load_breast_cancer'))

#step2
#from sklearn.model_selection import load_breast_cancer
#x_train,x_test,y_train,y_test=train_test_split(x,y,size=0/25,shuffle=true,random_state=42)

#step3
from sklearn.nighbors import KNeighborsClassification
model=KNeighborsClassification() 
hyperparametr(faraparametr) 
#bayad Ghabl az model begam

model=KNeighborsClassification(n_neigbors=1) 

#step4 ---- fitting
#model.fit(x_train,y_train)
model.fit(x,y)

#step5 ---- validating
y_pred=model.predict(x_train) 
train_score=accuracy_score(y_train,y_pred)
print(train_score) 

y_pred=model.predict(x_test) 
test_score=accuracy_score(y_true, y_pred) 
print(test_score)

from sklearn.dataset import load_breast_cancer
iris=load_iris()
x=iris.data 
print(iris.feature_names)
y=iris.target
print(iris.target_names)

#Decision tree
from sklearn.dataset import load_iris
iris=load_iris() 
x=iris.data
y=iris.target

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,size=0/25)

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(max_depth=1,random_state=42)
model.fit(x_train,y_train) 
y_pred=model.predict(x_train)
train_score=accuracy_score(y_true, y_pred) 
print(train_score) 
y_pred=model.predict(x_test) 
test_score=accuracy_score(y_true, y_pred)
print(test_score) 

#randomforest
from sklearn.datasets import load_iris
iris=load_iris() 
x=iris.datay=iris.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0/25,shuffle=True,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_train)
train_score=accuracy_score(y_train,y_pred)
print(train_score)

y_pred=model.predict(x_test)
test_score=accuracy_score(y_test,y_pred)
print(test_score)

#==============================================

x=iris.data
y=iris.target

from sklearn.model_selection import KFold

kf= KFold(n_splits=5,shuffle=True,random_state=42)
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()

from sklearn.model_selection import GridSearchCV

gs=GridSearchCV(model,cv=kf,scoring='accuracy')
gs.fit(x,y)
gs.best_score_

