import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
df = pd.read_csv('C:/Users/rakhmatulin/Desktop/Scotland/1.Machine_learning/1.Scikit-Learn_all_method_separate/iris_test.csv')

#1 take tata for graph
iris = datasets.load_iris()
data = pd.DataFrame(np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
print (data)
#2.Typical histogramm
fig, axs = plt.subplots(1, 2)
axs[0].hist(data['sepal length (cm)'],  len(data))
axs[0].set_title('sepal length')
axs[1].hist(data['petal length (cm)'],  len(data))
axs[1].set_title('petal length')
plt.show()

#3 two columns in one graph
fig, ax = plt.subplots(figsize=(40,5))
rects1 = ax.bar(np.arange(50), data['sepal width (cm)'][:50],   label='sepal width')
rects2 = ax.bar(np.arange(50), data['petal width (cm)'][:50],  label='petal width')

ax.set_ylabel('cm')
ax.set_xticks(np.arange(50))
ax.legend()
plt.show()

#4 seaborn распределение длин чашелистиков
sns_plot = sns.distplot(data['sepal width (cm)'])
#fig = sns_plot.get_figure()

#plt.plot(np.arange(50) ,data['sepal width (cm)'][:50])  # simply graph  
#plt.legend('ABCDEF', ncol=2, loc='upper left');
plt.show()
