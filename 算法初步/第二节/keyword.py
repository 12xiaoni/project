from config import *
import numpy as np
import re
import jieba
import os
import random

class TFIDF_KEY:
    def __init__(self):
        self.key_path = key_path
        self.aim_file_dict = {} # 存储目标文件的tf
        self.all_file_dict = dict() # 存储所有文件的字典的集合
        self.all_file_word = [] # 存储所有文件的未去重的词
        self.aim_file_tfidf = {} # 存储目标文件的tfidf
        self.aim_file_idf = {} # 目标文件在总文件中的逆文档频率

    def get_vob_fre(self): # 获得清洗过的数据
        news_list = os.listdir(self.key_path)
        self.aim_file = random.choice(news_list)
        for file_name in news_list:
            newspaper = open(self.key_path+'\\'+file_name,'r', encoding='utf8').read()
            newspaper = re.sub(r'[&gt, \u3000, &nbsp, \n, \d+, “, ”, \:]*', '', newspaper) #将无意义字符替换为空
            newspaper = re.sub(r'[\，, \。, \;, \?, \!, ？, 、, ．,（,）]', ' ', newspaper) #去掉文章中的标点符号
            cut_news = jieba.lcut(newspaper) # 使用结巴进行分词
            cut_news = [i for i in cut_news if i != ' '] # 去掉生成的空格
            self.all_file_word.append(cut_news)
        for i in range(len(self.all_file_word)):
            self.all_file_dict[i+1] = list(set(self.all_file_word[i]))
        # 对目标文件夹进行处理，统计词频
        aim_number = int(self.aim_file[3:4]) # 目标文件的序号
        self.aim_length = len(self.all_file_word[aim_number-1])
        for word in self.all_file_word[aim_number-1]:
            if word not in self.aim_file_dict:
                self.aim_file_dict[word] = 1
            else: self.aim_file_dict[word] += 1

    def fit(self):
        idf = 0 # 单个词的逆文档频率
        for word in self.aim_file_dict.keys():
            for oth in self.all_file_dict.values():
                if word in oth: idf += 1
            self.aim_file_idf[word] = np.log(idf/len(self.all_file_dict)) # 计算出目标文档每个字的逆文档频率
            idf = 0
        for word in self.aim_file_dict.keys():
            self.aim_file_tfidf[word] = abs(self.aim_file_dict[word]/self.aim_length*self.aim_file_idf[word]) # 计算目标文件每个词的tfidf
        key_words = sorted(zip(self.aim_file_tfidf.values(), self.aim_file_tfidf.keys()), reverse=True)[:10] # 获取目标文章关键词
        return key_words

if __name__ == '__main__':       
    t_k = TFIDF_KEY()
    t_k.get_vob_fre()
    t_k.fit()
    