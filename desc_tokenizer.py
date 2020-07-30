import pandas as pd
import numpy as np
import random

def feature_index(feature):
    # 针对每一个feature生成相应的index
    index = []
    if not (pd.isnull(feature)) and feature != 'N/A':
        for i in range((len(feature) - 1)):
            if feature[i].isalnum() != feature[i + 1].isalnum():
                if feature[i].isalnum():
                    index.append(1) #是字母或数字标为1
                else:
                    index.append(2) #否则标为2
            if not (feature[i].isalnum() or feature[i + 1].isalnum()):
                index.append(2)
        if len(index) == 0: #假如一个feature中没有符号出现，整个feature标为1
            if feature[-1].isalnum():
                index.append(1)
        else:
            if index[-1] == 2 and feature[-1].isalnum():
                index.append(1)
    return index


def feature_tokenize(feature):
    #将输入的feature进行分词
    tokenize = []
    word = ''
    if not(pd.isnull(feature)) and feature != 'N/A':
        for i in range(len(feature)-1):
            if not(feature[i].isalnum() and feature[i+1].isalnum()):
                word = word + feature[i]
                tokenize.append(word)
                word = ''
            else:
                word = word + feature[i]
        if len(tokenize) != 0 and not(feature[-1].isalnum()):
            tokenize.append(feature[-1])
        if len(tokenize) == 0:
            tokenize.append(feature)
    return tokenize

def description_index(feature,col):
    # 给一整个description生成index
    index_initial = []
    for i in range(3, 15):
        index_initial.append(feature_index(feature.iloc[col,i]))
    index_description = []
    for i in range(len(index_initial)):
        if index_initial[i] != []:
            index_description.append(index_initial[i])
            index_description.append([0]) #对无意义的符号（feature之间分隔符）标为'0'
    index_description = sum(index_description, [])
    index_description_new = [str(x) for x in index_description]
    return index_description_new

def description_generation_string(feature,col,delimiter):
    #拼接起数据文件中的feature，通过随机符号进行拼接
    des = ''
    for i in range(3,15):
        if not(pd.isnull(feature.iloc[col,i])) and feature.iloc[col,i] != 'N/A':
            des = des + feature.iloc[col,i]
            des = des + random.choice(delimiter)
    return des
