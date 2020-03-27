import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv('iris_test.csv')
df = df.values

x = df[:,0:4]
y = df[:,4]                                                                                           

from sklearn import preprocessing
normalized_x = preprocessing.normalize(x) #
#print (normalized_x)
standardized_x = preprocessing.scale(x)   # #print (standardized_x)

                                                      # Feature Selection и Feature Engineering
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(x, y)


                                                         #Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
# create the RFE model and select 3 attributes
rfe = RFE(model, 3)
rfe = rfe.fit(x, y)
# summarize the selection of the attributes


                                                           

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=42)
model.fit(x_train, y_train) 

#print(model)
# make predictions
expected = y_test
predicted = model.predict(x_test)


print ("ok")
# summarize the fit of the model
#print(metrics.classification_report(expected, predicted))
#print(metrics.confusion_matrix(expected, predicted))

from sklearn.metrics import accuracy_score  

print(accuracy_score(y_test, predicted))   



