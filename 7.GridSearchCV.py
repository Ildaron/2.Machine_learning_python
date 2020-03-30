import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv('train.csv').as_matrix()
#df = pd.read_csv('C:/Users/rakhmatulin/Desktop/Python/Kaggle/digit-recognizer/train_short.csv')
x = df[:,1:]
y = df[:,0:1]

from sklearn import preprocessing
normalized_x = preprocessing.normalize(x) 
standardized_x = preprocessing.scale(x)   
                                                      # Feature Selection и Feature Engineering
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
#model.fit(x, y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
# fit a CART model to the data
#model = RandomForestClassifier(n_estimators=100, min_samples_split=3)
#model.fit(x_train, y_train) 


#expected = y_test
#predicted = model.predict(x_test)

#from sklearn.metrics import accuracy_score  
#print ("ok")
#print(accuracy_score(y_test, predicted)) 


from sklearn.model_selection import GridSearchCV

min_samples_split_array=[3,4,5,6,7,8,10,20,30]
model = RandomForestClassifier()

grid = GridSearchCV(model, param_grid={'min_samples_split': min_samples_split_array})
grid.fit(x_train, np.ravel(y_train))
# knn.fit(X_train, np.ravel(y_train))

#best_cv_err = 1 - grid.best_score_
best = grid.best_estimator_.min_samples_split

print (best)

#print(grid)
# summarize the results of the grid search
#print(grid.best_score_)
#print(grid.best_params_)
print ("ok1")
#print(accuracy_score(y_test, predicted)) 
print ("ok2")
#print (grid.score(x_test,y_test))









