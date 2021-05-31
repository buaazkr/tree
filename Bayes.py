import sklearn.datasets as data
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

'''
    获取数据集，训练集与测试集之比为9：1
'''
sample = data.load_iris().data  # 数据集
label = data.load_iris().target  # 相应类别标签
sample_train, sample_test, label_train, label_test = train_test_split(sample, label, test_size=0.1, random_state=0)

'''
    利用贝叶斯网络进行训练
'''
clf = MultinomialNB()  # 建立朴素贝叶斯模型，训练并预测结果
clf.fit(sample_train, label_train)
pre = clf.predict(sample_test)

'''
    结果统计与计算
'''
print('实际结果')
print(label_test)
print('预测结果')
print(pre)
print('测试准确率')
N = len(pre)
wrong = 0
for i in range(N):
    if pre[i] != label_test[i]:
        wrong += 1
Accuracy=(N - wrong)/N
print(Accuracy)