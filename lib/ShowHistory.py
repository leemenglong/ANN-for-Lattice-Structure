# -*- coding: utf-8 -*-
# @Time    :2024/6/6 20:43
# @Author  :MENGLong Li
# @File    :ShowHistory.py
# @Software:PyCharm

import matplotlib.pyplot as plt


def plot_results(history, EPOCHS):
    # 显示 MAE
    plt.figure(figsize=(10, 6))
    epoch_range = range(1, EPOCHS + 1)
    plt.plot(epoch_range, history.history['mae'], label='train MAE')
    plt.plot(epoch_range, history.history['val_mae'], label='val MAE')
    plt.title('MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend(loc='best')
    plt.show()

    plt.figure(figsize=(10, 6))
    epoch_range = range(1, EPOCHS + 1)
    plt.plot(epoch_range, history.history['loss'], label='train loss')
    plt.plot(epoch_range, history.history['val_loss'], label='val loss')
    plt.title('LOSS')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.show()
    return 0
