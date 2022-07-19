import pandas as pd
import numpy as np
import os
import jieba
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier  

# TfidfVectorizer 是python中可以实现自动转换tfidf的一个类
# 下边的链接是关于tfidfvectorizer的一个简单的参数介绍
# https://blog.csdn.net/weixin_42462804/article/details/105433680?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-2.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-2.no_search_link

data = pd.read_csv(r'data\情感.csv').astype(str) # 不加astype(str) 会出现解码错误
 
def get_data():
    # 将原来的数据打乱，random.shuffle(),是random模块下随机打乱数据的函数
    datas = [data.iloc[i].to_list() for i in range(len(data))]
    random.shuffle(datas)
    # 划分训练集和测试集
    train_data = datas[:int(0.7*len(datas))]
    test_data = datas[int(0.7*len(datas)):]
    return train_data, test_data

def data_process(model_data):
    # input, target
    X, Y =[], []
    # 将每句话处理成tfidfvectorize可以接受的格式
    for i in range(len(model_data)):
        X.append(' '.join(jieba.lcut(model_data[i][1])))
        Y.append(model_data[i][0])
    return X, Y
        
train_data, test_data = get_data()

input, target = data_process(train_data)    
inputs, targets = data_process(test_data)

class KNN:
    def __init__(self, input, target, inputs, targets):
        self.train_input = input # 训练集测试集的input，target
        self.train_target = target
        self.test_input = inputs
        self.test_target = targets
        self.vector=TfidfVectorizer() # 实例化tfidf的一个类
        # self.result = [] # 存放最终的准确率
        
    # 训练数据的函数
    def tfidf(self, knn):
        # 将训练集测试集转换为tfidf的格式
        train_x = self.vector.fit_transform(self.train_input) 
        test_x = self.vector.transform(self.test_input)
        # 训练数据集
        knn.fit(train_x, self.train_target)
        #测试数据集
        y_pred = knn.predict(test_x)
        sum = 0
        for i in range(len(self.test_target)):
            if y_pred[i] == self.test_target[i]:
                sum+=1
        result = sum/len(targets)
        return result
    
    # 图表形式可视化
    def show_picture(self,result):
        x = np.array([i for i in range(1, 25)])
        y = np.array(result)
        plt.plot(x, y)
        plt.title('accuracy')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
        
        

if __name__ == '__main__':
    result = [] # 存放最终结果
    for i in tqdm(range(1, 25)):
        knn = KNeighborsClassifier(i) # 实例化knn的一个类
        k = KNN(input, target, inputs, targets)
        result.append(k.tfidf(knn))
    k.show_picture(result)

# vectorizer=TfidfVectorizer()  #定义了一个类的实例
# X=vectorizer.fit_transform(text)

