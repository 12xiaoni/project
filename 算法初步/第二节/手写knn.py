import pandas as pd
import numpy as np

data = pd.read_excel('class.xlsx')
aim_customer = np.array([4.3, 50, 0])
distance = [] # 存储目标点到其它点的距离
data_class = [] # 存储目标分类
index = [] # 存储前k个的索引

def KNN(k):
    for i in range(len(data)):
        x, y = data.iloc[i].to_list()[:3], data.iloc[i].to_list()[3:]
        # aim_customer 到其他点的欧式距离
        ath_distance = sum(pow((x - aim_customer),2)) 
        data_class.append(y[0])
        # 存入距离和索引,存入索引的目的是为了在排序后寻找到对应的y的分类
        distance.append((ath_distance, i)) 
    index =np.array([data_class[ind[1]] for ind in distance[:k]])
    # 求众数，得到最终的结果，np.bincount是计算次数的函数
    result = np.bincount(index).argmax() 
    print(f"第0类出现的次数{np.bincount(index)[0]}, 第一类出现的次数\
        {np.bincount(index)[1]},最终的结果:{result}")
    
if __name__ == '__main__':
    KNN(5)