# -*- coding: utf-8 -*-
# @Time    :2024/6/6 14:56
# @Author  :MENGLong Li
# @File    :MachineLearning.py
# @Software:PyCharm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential, layers, losses, utils
from lib.ShowHistory import plot_results
import warnings

warnings.filterwarnings("ignore")

def MachineLearningSet(train1 = 0, train2 = 0, train3 = 0, train4 = 0):

    # 加载数据
    path = "data\\"
    source_path = "source\\"
    x_file_name = path + "x_data.txt"
    young_modulus_data_name = path + "young_modulus_data.txt"
    yield_stress_data_name = path + "yield_stress_data.txt"

    '''
    --------------------------------------------------------------------------
    Yield Stress 训练
    --------------------------------------------------------------------------
    '''
    x_Data = np.loadtxt(x_file_name, delimiter=",")
    y_Data = np.loadtxt(yield_stress_data_name, delimiter=",")

    x_Data1 = []
    x_Data2 = []
    y_Data1 = []
    y_Data2 = []

    for i in range(y_Data.shape[0]):
        if y_Data[i] >= 55:
            x_Data1.append(x_Data[i, :])
            y_Data1.append(y_Data[i])
        else:
            x_Data2.append(x_Data[i, :])
            y_Data2.append(y_Data[i])

    x_Data1 = np.array(x_Data1)
    x_Data2 = np.array(x_Data2)
    y_Data1 = np.array(y_Data1)
    y_Data2 = np.array(y_Data2)

    # Machine Learning 参数设置
    EPOCHS = 100
    BATCH_SIZE = 32

    '''
    --------------------------------------------------------------------------
    Yield Stress 第1组训练
    --------------------------------------------------------------------------
    '''
    if train1 == 1:
        # 归一化
        x_data_scaler = StandardScaler().fit_transform(x_Data1)

        # 检验归一化之后，是否每列数据均值为0，标准差为1
        mean1 = np.mean(x_data_scaler[:, 0])
        std1 = np.std(x_data_scaler[:, 0])

        # 数据切分
        X_train, X_test, y_train, y_test = train_test_split(x_data_scaler, y_Data1, test_size=0.25, random_state=0)

        model_1 = Sequential([
            layers.Dense(units=64, activation='tanh', input_shape=[23]),
            layers.Dense(units=16, activation='tanh'),
            layers.Dense(1)
        ])
        model_1.summary()
        model_1.compile(optimizer='adam', loss='mse', metrics=['mae'])
        history1 = model_1.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test),
                               verbose=1)
        plot_results(history1,EPOCHS)
        model_1.save(source_path + 'Yield_Stress_group1_Model_1.h5')

        # 输出验证
        sample_1 = X_test[10:20, :]
        sample_1 = np.reshape(sample_1, (-1, 23))
        sample_1_label = y_test[10:20]
        sample_1_pred = model_1.predict(sample_1)
        print('Label:%s, Prediction:%s, Relative error: %s' % (
            sample_1_label, sample_1_pred, (sample_1_label - np.array(sample_1_pred).reshape(-1)) / sample_1_label))

        # 保存验证集数据
        data_y1 = model_1.predict(X_test)
        np.savetxt(source_path + 'pre_yield_stress_data1.csv', data_y1, delimiter=',')
        np.savetxt(source_path + 'label_yield_stress_data1.csv', y_test, delimiter=',')

    '''
    --------------------------------------------------------------------------
    Yield Stress 第2组训练
    --------------------------------------------------------------------------
    '''
    if train2 == 1:
        # 归一化
        x_data_scaler = StandardScaler().fit_transform(x_Data2)

        # 检验归一化之后，是否每列数据均值为0，标准差为1
        mean2 = np.mean(x_data_scaler[:, 0])
        std2 = np.std(x_data_scaler[:, 0])

        # 数据切分
        X_train, X_test, y_train, y_test = train_test_split(x_data_scaler, y_Data2, test_size=0.25, random_state=0)
        history2 = model_1.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test),
                               verbose=1)
        plot_results(history2,EPOCHS)
        model_1.save(source_path + 'Yield_Stress_group2_Model_1.h5')

        sample_2 = X_test[10:20, :]
        sample_2 = np.reshape(sample_2, (-1, 23))
        sample_2_label = y_test[10:20]
        sample_2_pred = model_1.predict(sample_2)
        print('Label:%s, Prediction:%s, Relative error: %s' % (
            sample_2_label, sample_2_pred, (sample_2_label - np.array(sample_2_pred).reshape(-1)) / sample_2_label))

        data_y2 = model_1.predict(X_test)
        np.savetxt(source_path + 'pre_yield_stress_data2.csv', data_y2, delimiter=',')
        np.savetxt(source_path + 'label_yield_stress_data2.csv', y_test, delimiter=',')

    '''
    --------------------------------------------------------------------------
    Young's modulus 训练
    --------------------------------------------------------------------------
    '''
    x_Data = np.loadtxt(x_file_name, delimiter=",")
    y_Data = np.loadtxt(young_modulus_data_name, delimiter=",")

    x_Data1 = []
    x_Data2 = []
    y_Data1 = []
    y_Data2 = []

    for iy in range(y_Data.shape[0]):
        if y_Data[iy] >= 12800 + (17000 - 12800) / 4000 * (iy + 1):
            x_Data1.append(x_Data[iy, :])
            y_Data1.append(y_Data[iy])
        else:
            x_Data2.append(x_Data[iy, :])
            y_Data2.append(y_Data[iy])

    # plt.figure()
    # plt.scatter(np.linspace(0, np.size(y_Data1, 0) - 1, np.size(y_Data1, 0)), y_Data1, c='blue', label='data1')
    # plt.scatter(np.linspace(0, np.size(y_Data2, 0) - 1, np.size(y_Data2, 0)), y_Data2, c='red', label='data2')
    # plt.show()

    x_Data1 = np.array(x_Data1)
    x_Data2 = np.array(x_Data2)
    y_Data1 = np.array(y_Data1)
    y_Data2 = np.array(y_Data2)

    # Machine Learning 参数设置
    EPOCHS = 100
    BATCH_SIZE = 32

    '''
    --------------------------------------------------------------------------
    Young Modulus 第1组训练
    --------------------------------------------------------------------------
    '''
    if train3 == 1:
        # 归一化
        x_data_scaler = StandardScaler().fit_transform(x_Data1)

        # 检验归一化之后，是否每列数据均值为0，标准差为1
        mean3 = np.mean(x_data_scaler[:, 0])
        std3 = np.std(x_data_scaler[:, 0])

        # 数据切分
        X_train, X_test, y_train, y_test = train_test_split(x_data_scaler, y_Data1, test_size=0.25, random_state=0)

        model_2 = Sequential([
            layers.Dense(units=1024, activation='relu', input_shape=[23]),
            layers.Dropout(0.2),
            layers.Dense(units=1024, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(units=1024, activation='relu'),
            layers.Dense(units=512, activation='relu'),
            layers.Dense(1)
        ])
        model_2.summary()
        model_2.compile(optimizer='adam', loss='mse', metrics=['mae'])
        history3 = model_2.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test),
                               verbose=1)
        plot_results(history3,EPOCHS)
        model_2.save(source_path + 'Young_Modulus_group1_Model_1.h5')

        sample_1 = X_test[10:20, :]
        sample_1 = np.reshape(sample_1, (-1, 23))
        sample_1_label = y_test[10:20]
        sample_1_pred = model_2.predict(sample_1)
        print('Label:%s, Prediction:%s, Relative error: %s' % (
            sample_1_label, sample_1_pred, (sample_1_label - np.array(sample_1_pred).reshape(-1)) / sample_1_label))

        data_y1 = model_2.predict(X_test)
        np.savetxt(source_path + 'pre_young_modulus_data1.csv', data_y1, delimiter=',')
        np.savetxt(source_path + 'label_young_modulus_data1.csv', y_test, delimiter=',')

    '''
    --------------------------------------------------------------------------
    Young Modulus 第2组训练
    --------------------------------------------------------------------------
    '''
    if train4 ==1:
        # 归一化
        x_data_scaler = StandardScaler().fit_transform(x_Data2)

        # 检验归一化之后，是否每列数据均值为0，标准差为1
        mean4 = np.mean(x_data_scaler[:, 0])
        std4 = np.std(x_data_scaler[:, 0])

        # 数据切分
        X_train, X_test, y_train, y_test = train_test_split(x_data_scaler, y_Data2, test_size=0.25, random_state=0)

        model_3 = Sequential([
            # layers.Dense(units=1024, activation='relu', input_shape=[23]),
            # layers.Dropout(0.2),
            # layers.Dense(units=1024, activation='relu'),
            # layers.Dropout(0.2),
            # layers.Dense(units=1024, activation='relu'),
            # layers.Dense(units=512, activation='relu'),
            # layers.Dense(1)
            layers.Dense(units=512, activation='relu', input_shape=[23]),
            layers.Dense(units=256, activation='relu'),
            layers.Dense(1)
        ])
        model_3.summary()
        model_3.compile(optimizer='adam', loss='mse', metrics=['mae'])
        history4 = model_3.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test),
                               verbose=1)
        plot_results(history4,EPOCHS)
        model_3.save(source_path + 'Young_Modulus_group2_Model_3.h5')

        sample_1 = X_test[10:20, :]
        sample_1 = np.reshape(sample_1, (-1, 23))
        sample_1_label = y_test[10:20]
        sample_1_pred = model_3.predict(sample_1)
        print('Label:%s, Prediction:%s, Relative error: %s' % (
            sample_1_label, sample_1_pred, (sample_1_label - np.array(sample_1_pred).reshape(-1)) / sample_1_label))

        data_y1 = model_3.predict(X_test)
        np.savetxt(source_path + 'pre_young_modulus_data2.csv', data_y1, delimiter=',')
        np.savetxt(source_path + 'label_young_modulus_data2.csv', y_test, delimiter=',')
