# -*- coding: utf-8 -*-
# @Time    :2024/6/6 20:23
# @Author  :MENGLong Li
# @File    :main.py
# @Software:PyCharm

import numpy as np
from lib.ReadTxtDataFromOdbFile import ReadTxtDataFromOdbFile
from lib.MachineLearning import MachineLearningSet


"-------------------------------------------------- Step 1 ------------------------------------------------------------"
"采用 lib 文件中的 AbaqusModelCreate.py; 在ABAQUS软件中运行该脚本生成指定结构"

"-------------------------------------------------- Step 2 ------------------------------------------------------------"
"采用 lib 文件中的 SubmitInpFile.py; 在计算服务器上运行该程序来批量提交ABAQUS计算"

"-------------------------------------------------- Step 3 ------------------------------------------------------------"
"采用 lib 文件中的 ReadOdbFile; 运行该脚本获得abaqus计算之后ODB文件，获取其中原始数据"

"-------------------------------------------------- Step 4 ------------------------------------------------------------"
"运行下列程序，来进行数据的机器学习"

if __name__ == '__main__':
    num_group = 4
    num_index = 1000
    LengthX = 4
    LengthY = 4
    LengthZ = 4

    '''
    是否重新初始化训练数据集输入?
    set_ini = 1 >> 重新初始化
    set_ini = 0 >>不初始化
    按需更改
    '''
    set_ini = 0
    print('|\n|\n')
    if set_ini:
        print('Beginning to initialize the data set for Machine Learning')
        ReadTxtData = ReadTxtDataFromOdbFile(num_group, num_index, LengthX, LengthY, LengthZ)
        ReadTxtData.run()
        print('|\n|\n')
        print('Finished data initialization')
    else:
        print('Skip the initialization')


    '''
        是否重新训练?
        Train1 = 1 or 0 >> 重新训练 or 不重新训练
        Train2 = 1 or 0 >> 重新训练 or 不重新训练
        Train3 = 1 or 0 >> 重新训练 or 不重新训练
        Train4 = 1 or 0 >> 重新训练 or 不重新训练
        按需更改
        '''
    print('|\n|\n')
    print('Beginning to Machine Learning')
    MachineLearningSet(train1=0, train2=0, train3=0, train4=0)
    print('|\n|\n')
    print('Finished Machine Learning')


"-------------------------------------------------- Step 5 ------------------------------------------------------------"
"机器学习过之后的数据在 source 文件夹中，为csv数据形式，需要进行进一步分析以获得文章的中数据"
