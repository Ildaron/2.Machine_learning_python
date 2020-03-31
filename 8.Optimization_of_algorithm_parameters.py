import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv('iris_test.csv')


df = df.values
#names = ['first', 'second', 'third', 'fouth', 'result']   
#df = np.vstack([names, df])
x = df[:,0:4]
y = df[:,4]                                                                                           

from sklearn import preprocessing
normalized_x = preprocessing.normalize(x) 
standardized_x = preprocessing.scale(x)   

                                                      # Feature Selection и Feature Engineering
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(x, y)
#print(model.feature_importances_)

                                                         # Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
# create the RFE model and select 3 attributes
rfe = RFE(model, 3)
rfe = rfe.fit(x, y)
                                        



from sklearn.metrics import accuracy_score  

#print(accuracy_score(y_test, predicted))   # проверяем точность

from sklearn import metrics
from sklearn.svm import SVC
# fit a SVM model to the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
model = SVC()

model.fit(x_train, y_train) 
#print(model)
# make predictions
expected = y_test
predicted = model.predict(x_test)
# summarize the fit of the model
#print(metrics.classification_report(expected, predicted))
#print(metrics.confusion_matrix(expected, predicted))
from sklearn.metrics import accuracy_score  
print ("ok")
print(accuracy_score(y_test, predicted)) 


from sklearn.model_selection import GridSearchCV


param_grid = {'C' : [0.01, 0.1, 1, 10],'kernel': ['rbf', 'linear'],}

grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(x_train, y_train)
#print(grid)
# summarize the results of the grid search
#print(grid.best_score_)
#print(grid.best_params_)
print ("ok")
print(accuracy_score(y_test, predicted)) 
print ("ok1")
print (grid.score(x_test,y_test))



from sklearn.model_selection import GridSearchCV
param_grid = {'C' : [0.01, 0.1, 1, 10],'kernel': ['rbf', 'linear'],}
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(x_train, y_train)
#print(grid)
# summarize the results of the grid search
#print(grid.best_score_)
#print(grid.best_params_)
print ("ok")
print(accuracy_score(y_test, predicted)) 
print ("ok1")
print (grid.score(x_test,y_test))



