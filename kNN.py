from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify(inX, dataSet, labels, k):
    # inX为待测数据，表示为一个向量；dataSet为数据集，表示为向量数组；labels为标签；k为取前k个最近距离元素
    dataSetSize = dataSet.shape[0]  # shape是array的属性，描述了一个数组的形状，就是它的维度。所以dataset.shape[0]就是样本集的个数
    diff_mat = tile(inX, (dataSetSize, 1)) - dataSet  # 用欧式距离公式计算两个向量点之间的距离
    sq_diff_mat = diff_mat ** 2  # 平方
    sq_distances = sq_diff_mat.sum(axis=1)  # 对行求和，若axis=0是对列求和
    distances = sq_distances ** 0.5  # 开根号
    sorted_dist_indicies = distances.argsort()  # 对数组升序排序,argsort()是对位置进行排序！！！
    classCount = {}  # classCount是字典，key是标签
    for i in range(k):  # 选择距离最小的k个点
        voteIlabel = labels[sorted_dist_indicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        # dict.get(key, default=None)函数，key就是dict中 1的键voteIlabel，如果不存在则返回一个0并存入dict，如果存在则读取当前值并+1；
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 进行降序排序
    # py3中没有iteritems，直接用items，py2则用iteritems
    return sortedClassCount[0][0]  # sortedClassCount[0][0]也就是排序后的次数最大的那个label

	
	# 将约会数据文本记录转化为numpy的解析程序
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()  # readlines()自动将文件内容分析成一个行的列表，该列表可以由 for ... in ... 结构进行处理。
    numberOfLines = len(arrayOLines)  # 得到文件的行数
    # 创建返回Numpy的矩阵
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    # 解析文件数据到列表
    for line in arrayOLines:
        line = line.strip()  # 移除字符串头尾指定的字符序列
        listFromLine = line.split('\t')  # 截取所有的回车字符
        returnMat[index, :] = listFromLine[0:3]  # 选取前三列
        classLabelVector.append(int(listFromLine[-1]))  # 将列表的最后一列存储到向量classLabelVector中
        index += 1
    return returnMat, classLabelVector