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

                                                         #Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
# create the RFE model and select 3 attributes
rfe = RFE(model, 3)
rfe = rfe.fit(x, y)
# summarize the selection of the attributes
#print(rfe.support_)   # первый тип откидывает так как в print(model.feature_importances_) видно что у него самый маленькое значение

                                                


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)


from sklearn.metrics import accuracy_score  

#print(accuracy_score(y_test, predicted))   # проверяем точность

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
# fit a CART model to the data
model = DecisionTreeClassifier()
                                  # class sklearn.ensemble.RandomForestClassifier(n_estimators=10,
                                  #                                              criterion='gini', 
                                  #                                               max_depth=None,
                                  #                                               min_samples_split=2,
                                  #                                               min_samples_leaf=1, 
                                  #                                               min_weight_fraction_leaf=0.0, 
                                  #                                               max_features='auto', 
                                  #                                               max_leaf_nodes=None, 
                                  #                                               bootstrap=True, 
                                  #                                               oob_score=False,
                                  #                                               n_jobs=1, 
                                  #                                               random_state=None,
                                  #                                               verbose=0, 
                                  #                                               warm_start=False, 
                                  #                                               class_weight=None)


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

