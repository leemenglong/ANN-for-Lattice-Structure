# -*- coding: utf-8 -*-
# @Time    :2024/6/6 10:51
# @Author  :MENGLong Li
# @File    :SubmitInpFile.py
# @Software:PyCharm

from abaqus import *
from abaqusConstants import *
import os

num_group = 4  # 内部节点个数为1-4个
num_index = 1000  # 每个节点随机生成1000个运算案例

os.chdir(r"F:\\MaterialsMachineLearning\\abaqus\\Job-BJX-3\\")  # InpFile文件存在的绝对路径

for group in range(1, num_group + 1):
    for i in range(num_index / 40):  # 每40一组进行文件提交运算
        nameList = []
        inpNameList = []
        for j in range(40):
            n = i * 40 + j
            name = 'Job-BJX-%s-No%s' % (n, group)
            inpName = '%s.inp' % name
            nameList.append(name)
            inpNameList.append(inpName)
        for j in range(40):
            mdb.JobFromInputFile(name=nameList[j], inputFileName=inpNameList[j], numCpus=1, numDomains=1)
            mdb.jobs[nameList[j]].submit()
        for j in range(40):
            mdb.jobs[nameList[j]].waitForCompletion()
