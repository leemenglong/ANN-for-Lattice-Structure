# ANN-for-Lattice-Structure
This is the data set and the python & Matlab code for  *Prediction of Mechanical Properties of Lattice Structures: an Application of Artificial Neural Networks Algorithms*


### 文件夹框架

MachineLearningANN

|__ data 机器学习之前所有txt格式的数据文件

|__ lib 静态库

​        |__  AbaqusModelCreate.py  ABAQUS模型批量创建

​        |__  MachineLearning.py  ANN机器学习

​        |__  ReadOdbFile.py  获取ABAQUS计算过后的结果文件（ODB）中的原始数据

​        |__  ReadTxtDataFromOdbFile.py  读取原始数据文件，将“杨氏模量”“屈服强度”汇总，生成待学习文件

​        |__  ShowHistory.py  显示机器学习过程中的损失函数和mae值

​        |__  SubmitInpFile.py  批量提交Inp文件，以供ABAQUS计算

|__ source 结果数据

​        |__ *.h5 对应机器学习已经训练完成模型

​        |__ *.csv 对应机器学习原始（label）数据与预测数据（pre）

|__ main.py 主要运行文件

|__ Readme.md 程序说明

