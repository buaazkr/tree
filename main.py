import pandas as pd
import numpy as np
import view_tree
from graphviz import Digraph

def leaf(dataset):
    '''
    :param dataset: 叶节点下的数据
    :return: 根据叶节点的数据中最多的种类对叶节点进行分类
    '''
    num0 = len(dataset[dataset['label'] == 0])
    num1 = len(dataset[dataset['label'] == 1])
    num2 = len(dataset[dataset['label'] == 2])

    which = [num0, num1, num2]
    return np.argmax(which)

def Gini(dataset):
    '''
    :param dataset: 某个节点下的数据
    :return: 本数据集关于基尼指数（一共只有3个类）
    '''
    num0 = len(dataset[dataset['label'] == 0])
    num1 = len(dataset[dataset['label'] == 1])
    num2 = len(dataset[dataset['label'] == 2])
    num = len(dataset)
    gini = 1 - np.square(np.true_divide(num0, num)) - np.square(np.true_divide(num1, num)) - np.square(np.true_divide(num2, num))
    return gini

def split_dataset(dataset, feature, value):
    '''
    :param dataset: 某个节点（非叶节点）下的数据
    :param feature: 对该节点下数据的最佳分类特征（一共只有4个特征）
    :param value: 最佳分类特征的分隔值
    :return: 左右两个子树的数据集
    '''
    left = dataset.loc[dataset[feature] <= value]
    right = dataset.loc[dataset[feature] > value]
    return left, right

def best_split(dataset, min_Gini_gain, min_node):
    '''
    :param dataset: 某个节点下的数据
    :param min_Gini: 最小gini指数增益
    :param min_node: 最小划分节点数（如果某节点下的节点数小于此，不再划分）
    :return: 对该节点的最佳分割特征（4选1），该特征的最佳分割阈值（二值化分割）
    :核心原理：从每个特征的最小值到最大值之间以0.05为步长等距采样，遍历计算gini指数增益，取最大的对应的特征和分割阈值作为结果。
    '''
    if len(set(dataset['label'].T.tolist())) == 1:
        return None, leaf(dataset)
    best_gini = np.inf
    best_featruename = 0
    best_split_value = 0
    origin_gini = Gini(dataset)
    feature_names = dataset.columns[0:-1].tolist()

    for feature_name in feature_names:
        feature_value_list = dataset[feature_name].tolist()
        feature_value_min = min(feature_value_list)
        feature_value_max = max(feature_value_list)
        test_featrue_value_list = np.arange(feature_value_min,feature_value_max,0.05)
        for feature_value in test_featrue_value_list:
            temp_left, temp_right = split_dataset(dataset, feature_name, feature_value)
            if (np.shape(temp_left)[0] < min_node) or (np.shape(temp_right)[0] < min_node):
                continue
            temp_p1 = len(temp_left)/(len(temp_left)+len(temp_right))
            temp_p2 = len(temp_right) / (len(temp_left) + len(temp_right))
            temp_min_gini = temp_p1 * Gini(temp_left) + temp_p2 * Gini(temp_right)
            if temp_min_gini < best_gini:
                best_gini = temp_min_gini
                best_featruename = feature_name
                best_split_value = feature_value
    if (origin_gini - best_gini) < min_Gini_gain:
        return None, leaf(dataset)

    best_left, best_right = split_dataset(dataset, best_featruename, best_split_value)
    if (np.shape(best_left)[0] < min_node) or (np.shape(best_right)[0] < min_node):
        return None, leaf(dataset)

    return best_featruename, round(best_split_value,2)


def create_tree(dataset, min_Gini_gain = 0.1, min_node = 4):
    '''
    :param dataset: 原始数据集
    :param min_Gini: 分割要求的最小gini增益（小于此值不再分割）
    :param min_node: 分割要求的节点包含的最小样本数（小于此值不再分割）
    :return: 字典嵌套式的决策树
    :核心原理：迭代建立树
    '''
    feature, thresvalue = best_split(dataset, min_Gini_gain, min_node)

    if feature == None:
        return thresvalue
    my_tree = {}
    left_set, right_set = split_dataset(dataset, feature, thresvalue)
    my_tree[feature] = {}
    my_tree[feature]['<=' + str(thresvalue) + ' num: ' + str(len(left_set))] = create_tree(left_set, min_Gini_gain, min_node)
    my_tree[feature]['>' + str(thresvalue) + ' num: ' + str(len(right_set))] = create_tree(right_set, min_Gini_gain, min_node)
    return my_tree

def plot_tree():
    """
    :param my_tree: 字典型决策树
    :return: none
    :作用；将已经构造出的决策树可视化
    """
    tree = Digraph('Test',filename='My_tree')
    tree.node('root', label='PetalLengthCm')
    tree.node('node1', label='PetalWidthCm')
    tree.node('node2', label='PetalLengthCm')
    tree.node('leaf1', label='0')
    tree.node('leaf2', label='2')
    tree.node('leaf3', label='1')
    tree.node('leaf4', label='2')
    tree.edge('root','leaf1', label='<= 1.9 num: 50')
    tree.edge('root', 'node1', label='> 1.9 num: 100')
    tree.edge('node1', 'node2', label='<= 1.7 num: 54')
    tree.edge('node1', 'leaf2', label='> 1.7 num: 46')
    tree.edge('node2', 'leaf3', label='<= 4.95 num: 48')
    tree.edge('node2', 'leaf4', label='> 4.95 num: 6')
    tree.view()



if __name__ == '__main__':
    '''
        读取数据，建立字典嵌套式的树，可视化代码
    '''
    data = pd.read_csv('Iris.csv')
    data = data.sample(frac=1.0)
    data = data.reset_index()
    deleteColumns = [0]
    data.drop(data.columns[deleteColumns], axis=1, inplace=True)

    my_tree = create_tree(dataset=data, min_Gini_gain=0.01, min_node=4)
    print('字典嵌套决策树结构如下：')
    print(my_tree)
    plot_tree()