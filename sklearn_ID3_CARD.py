import pandas as pd
from sklearn import tree
from matplotlib import pyplot as plt

'''
    获取数据
'''
data = pd.read_csv('Iris.csv')
data.head()
sample = data.drop(['label'],axis=1)
label = data.loc[:,'label']

'''
    用户输入构造决策树的方法，CARD法-输入‘gini’,ID3法输入‘entropy’
'''
method = input('Which method to struct a tree, please input "gini" or "entropy"  ')
dc_tree = tree.DecisionTreeClassifier(criterion=method,min_samples_leaf=5,min_samples_split=5,min_impurity_decrease=0.01)
#决策树分裂出来的叶子最少要有5个样本，如果再往下分发现少于5个样本节点就没有必要往下分了
dc_tree.fit(sample,label)

'''
    结果显示
'''
label_predict = dc_tree.predict(sample)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(label,label_predict)
print(accuracy)

fig1 = plt.figure(figsize=(8,8))
tree.plot_tree(dc_tree,filled="True",feature_names=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'],class_names=['setosa','versicolor','virginica'])
plt.show()