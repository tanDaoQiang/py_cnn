#coding=utf-8
#Version:python3.6.0
_date_ = '2019/7/10 15:58'
_author_ = 'tan'

import  numpy as np
import matplotlib.pyplot as plt

seed = 2
def generateds():
    #基于seed产生的随机数
    rmd = np.random.RandomState(seed)
    #随机数返回300行2列的矩阵， 表示300组坐标点(x0,x1)作为输入数据集
    X = rmd.rand(300,2)
    #从X这个300行2列的矩阵中取出一行，判断如果两个坐标的平方和小于2 给Y赋值1，其余赋值0
    Y_ = [int(x0*x0 + x1*x1 <2) for (x0, x1) in X]
    #遍历Y中的每个元素， 1赋值red，其余赋值blue
    Y_c = [['red' if y else 'blue'] for y in Y_]
    #对数据集X和标签Y进行形状整理，第一个元素-1表示跟随第二列计算，第二个元素表示多少列，可见X为两列，Y为1列
    X = np.vstack(X).reshape(-1, 2)
    Y_ = np.vstack(Y_).reshape(-1, 1)

    return X, Y_, Y_c