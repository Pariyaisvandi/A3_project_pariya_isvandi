#project 3 
#pariya_isvandi

from sklearn.datasets import load_breast_cancer
data=load_breast_cancer()
x=data.data
y=data.target

data.feature_names

data.target_names

#==============================================================

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
#0.9419810588417947

print(gs.best_params_)
#{'metric': 'manhattan', 'n_neighbors': 5, 'weights': 'uniform'}

#===============================================================
#DecisionTree

from sklearn.model_selection import KFold
kf= KFold(n_splits=5,shuffle=True,random_state=42)

#-----MODEL SELECTION
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
my_params= { 'n_neighbors':[3,5,7,8,9],
            'metric':['minkowski'  , 'euclidean' , 'manhattan'],
           'weights':['uniform','distance']}

from sklearn.datasets import load_iris
iris=load_iris() 
x=iris.data
y=iris.target 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25, shuffle=True, random_state=42)

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(max_depth=1,random_state=42)

model.fit(x_train,y_train)

#===========================================================
#medonam dorost nist chon error mizane

y_pred=model.predict(x_train)
train_score=accuracy_score(y_train,y_pred)
print(train_score)

y_pred=model.predict(x_test)
test_score=accuracy_score(y_test,y_pred)
print(test_score) 

#==========================================================
#RandomForest
from sklearn.model_selection import KFold
kf= KFold(n_splits=5,shuffle=True,random_state=42)

#-----MODEL SELECTION
from sklearn.datasets import load_iris
iris=load_iris()

x=iris.data
y=iris.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25, shuffle=True, random_state=42)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(random_state=42,n_estimators=40)

model.fit(x_train,y_train)

#==========================================================
#rror mizne baram
y_pred=model.predict(x_train)
train_score=accuracy_score(y_train,y_pred)
print(train_score) 

y_pred=model.predict(x_test)
test_score=accuracy_score(y_test,y_pred)
print(test_score) 

#===========================================================
#SVR
from sklearn.model_selection import KFold
kf= KFold(n_splits=5,shuffle=True,random_state=42)

from sklearn.svm import SVR
model=SVR()

my_params= { 'n_neighbors':[3,5,7,8,9],
            'metric':['minkowski'  , 'euclidean' , 'manhattan'],
           'weights':['uniform','distance']}


from sklearn.model_selection import GridSearchCV

gs=GridSearchCV(model,my_params,cv=kf,scoring='accuracy')
gs.fit(x,y)

print(gs.best_score_)

print(gs.best_params_)


#===========================================================
