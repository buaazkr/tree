import pandas as pd
import numpy as np
from sklearn import tree
from matplotlib import pyplot as plt
data = pd.read_csv('Iris.csv')
data.head()

X = data.drop(['label'],axis=1)
y = data.loc[:,'label']
print(X.shape,y.shape)#打印X，y的维度

dc_tree = tree.DecisionTreeClassifier(criterion='gini',min_samples_leaf=5,min_samples_split=5,min_impurity_decrease=0.01)
#决策树分裂出来的叶子最少要有5个样本，如果再往下分发现少于5个样本节点就没有必要往下分了
dc_tree.fit(X,y)

y_predict = dc_tree.predict(X)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y,y_predict)
print(accuracy)

fig1 = plt.figure(figsize=(8,8))
tree.plot_tree(dc_tree,filled="True",feature_names=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'],class_names=['setosa','versicolor','virginica'])
#填充底色 分类名称
plt.show()