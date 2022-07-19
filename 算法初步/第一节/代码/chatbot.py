import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 

CORPUS_PATH = r'C:\Users\Administrator\Desktop\算法初步\第一节\data\conversation_test.txt'
class CorpusSearch:
    def __init__(self):
        pass
    def load_seq(self):
        question = []
        answer = []
        math = 0
        for i in open(CORPUS_PATH, encoding='utf8').readlines():
            i = i.strip()
            math += 1
            if math%2 == 1:
                words = [word for word in i]
                question.append(words)
            else:
                answer.append(i)
        return question,answer

    def que(self):
        questions = []
        ques,_ = self.load_seq()
        questions = [' '.join(word)for word in ques]
        return questions 

    def tfidfs(self,word):

        questions = self.que()
        # 每一句话的向量长度是295，总共有232句话
        vector = TfidfVectorizer(analyzer='char')
        vectors = vector.fit(questions)
        vects = vectors.transform(questions)
        vect = vectors.transform(word)
        maxs = []
        for word in vects:
            maxs.append(cosine_similarity(vect, word))
        maxs = np.array(maxs)
        return np.argmax(maxs)

    def search_answer(self,question):
        sen = []
        for i in question:
            sen.append(i)
        word = [' '.join(sen)]
        q,a = self.load_seq()
        return a[self.tfidfs(word)]


if __name__ == '__main__':
    cor = CorpusSearch()
    while 1:
        print(cor.search_answer(input('请输入->')))