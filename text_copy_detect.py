import numpy as np 
import pandas as pd 
import jieba
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import pickle,os

#step1:数据加载
#加载停用词
with open('chinese_stopwords.txt','r',encoding = 'utf-8') as file:
	#print(file.readlines())
	stopwords = [i[:-1] for i in file.readlines()]
# print(stopwords)
#加载数据
news = pd.read_csv('sqlResult.csv',encoding = 'gb18030')
# print(news.shape)
# print(news.head())

#step2:数据预处理
#1）数据清洗，针对content字段为空的情况，进行dropna
news = news.dropna(subset = ['content'])
# print(news.shape)

#2)分词，使用jieba进行分词
def split_text(text):
	text = text.replace(' ','')
	text = text.replace('\n','').replace('\r','')
	text2 = jieba.cut(text.strip())
	result = ' '.join([w for w in text2 if w not in stopwords])
	return result

# print(news.iloc[0].content)
# print(split_text(news.iloc[0].content))

#3)将处理好的分词保存到corpus.pkl,方便下次调用
if not os.path.exists('corpus.pkl'):
	corpus = list(map(split_text,[str(i) for i in news.content]))
	print(corpus[0])
	with open('corpus.pkl','wb') as file:
		pickle.dump(corpus,file)
else:
	#调用上次处理结果
	with open('corpus.pkl','rb') as file:
		corpus = pickle.load(file)

#step3:提取文本特征TF-IDF
countvectorizer = CountVectorizer(encoding = 'gb18030',min_df=0.015)
tfidftransformer = TfidfTransformer()

countvector = countvectorizer.fit_transform(corpus)
tfidf = tfidftransformer.fit_transform(countvector)

#step4:预测文章风格是否和自己一致
# 标记是否为自己的新闻
label = list(map(lambda source:1 if '新华' in str(source) else 0,news.source))

# 数据集切分
X_train,X_test,y_train,y_test = train_test_split(tfidf.toarray(),label,test_size = 0.3,random_state = 33)

#使用朴素贝叶斯进行训练
clf = MultinomialNB()
clf.fit(X_train,y_train)
#在测试集上预测
prediction = clf.predict(X_test)
#labels = np.array(label)
compare_news_index = pd.DataFrame({'prediction':prediction,'labels':y_test})
# print(compare_news_index)
# step5:找到可能的copy文章，即预测为label ==1 ，实际label == 0
copy_news_index = compare_news_index[(compare_news_index['prediction'] == 1) & (compare_news_index['labels'] == 0)].index
print("预测抄袭文章id:\n",copy_news_index)

#step6:根据模型预测的结果来对全量文本进行比对，数量很大，可先采用k-means进行聚类降维。
if not os.path.exists("label.pkl"):
	normalizer = Normalizer()
	scaled_array = normalizer.fit_transform(tfidf.toarray())
	kmeans = KMeans(n_clusters = 18)
	k_labels = kmeans.fit_predict(scaled_array)
	print(k_labels)
	print(len(k_labels))
	#保存到文件，方便下次使用
	with open('label.pkl','wb') as file:
		pickle.dump(k_labels,file) 
else:
	with open("label.pkl","rb") as file:
		k_labels = pickle.load(file)
#id:对应每篇文章的id,class对应聚类后得到的分类，0-17类
if not os.path.exists("id_class.pkl"):
	id_class = {index:class_ for index,class_ in enumerate(k_labels)}
	with open('id_class.pkl','wb') as file:
		pickle.dump(id_class,file)
else:
	with open("id_class.pkl","rb") as file:
		id_class = pickle.load(file)
#统计新华社发布文章的index
if not os.path.exists("class_id.pkl"):	
	#实际为新华社的新闻
	news["labels"] = label
	xinhuashe_news_index = news[news['labels'] == 1].index
	class_id = defaultdict(set)
	for index,class_ in id_class.items():
		#只统计新华社发布的class_id
		if index in set(xinhuashe_news_index):
			class_id[class_].add(index)
	with open('class_id.pkl','wb') as file:
		pickle.dump(class_id,file)
else:
	with open("class_id.pkl","rb") as file:
		class_id = pickle.load(file)

#step7:找到一篇可能的copy文章，从相同的label中，找到对应的新华社文章，并按照tfidf相似度矩阵从大到小排列，取top10
def find_similar_text(cpindex,top = 10):
	#只在新华社发布的文章中查找和疑似抄袭文章同一聚类内的文章id
	dist_dict = {i:cosine_similarity(tfidf[cpindex],tfidf[i]) for i in class_id[id_class[cpindex]]}
	return sorted(dist_dict.items(),key = lambda x:x[1][0],reverse = True)[:top]

#在copy_news_index里面找一个
cpindex = copy_news_index[0]
similar_list = find_similar_text(cpindex)
print("相似新华社文章id列表",similar_list)
print('怀疑抄袭\n',news.iloc[cpindex].content)

#找到最相似原文
similar2 = similar_list[0][0]
print('相似原文\n',news.iloc[similar2].content)