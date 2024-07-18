# -*- coding: utf-8 -*-
# @Time    :2024/6/6 11:04
# @Author  :MENGLong Li
# @File    :ReadTxtDataFromOdbFile.py
# @Software:PyCharm

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os


class ReadTxtDataFromOdbFile:
    def __init__(self, num_group, num_index, LengthX, LengthY, LengthZ):
        self.num_group = num_group
        self.num_index = num_index
        self.LengthX = LengthX
        self.LengthY = LengthY
        self.LengthZ = LengthZ
        self.path = "data\\"

    def run(self):
        Aero = self.LengthX * self.LengthZ

        str_list = ["i, Noi, Young's Modulus, Yield Stress"]
        data_list = []

        for index in range(1, self.num_group + 1):

            for i in range(self.num_index):
                filename = self.path + "Data-BJX-%s-No%s.txt" % (i, index)

                data = np.genfromtxt(filename, delimiter=',')
                data_Del = np.delete(data, 2, 1)
                strain = -data_Del[:, 0] / self.LengthY
                stress = -data_Del[:, 1] / Aero
                young_module = stress / strain

                yield_stress_cal = (strain - 0.002) * young_module[1]

                # 寻找0.002对应交点
                jian = yield_stress_cal - stress
                jian_Data = interp1d(jian, strain, kind='linear')
                jian_Data.y[1] = jian_Data.y[0]
                jian_Data.y[0] = 0.0
                yield_strain = jian_Data(0)
                yield_stress_data = interp1d(strain, yield_stress_cal, kind='linear')
                yield_stress = yield_stress_data(yield_strain)

                str_list.append("%s, %s, %s, %s" % (i, index, young_module[1], float(yield_stress)))
                np_array = np.array([i, index, young_module[1], float(yield_stress)])
                data_list.append(np_array)

        np_array = np.array(data_list)  # "i, Noi, Young's Modulus, Yield Stress"

        data_young_modulus = np_array[:, 2]
        data_yield_stress = np_array[:, 3]

        # 获取 machine learning 的 x 输入
        data_input = np.zeros((self.num_group * self.num_index, 23))
        for index in range(1, self.num_group + 1):
            for i in range(self.num_index):
                element_file = self.path + "Job-BJX-%s-No%s-element.txt" % (i, index)
                location_file = self.path + "Job-BJX-%s-No%s-Location.txt" % (i, index)
                element = np.genfromtxt(element_file, delimiter=',').astype('int')
                location = np.genfromtxt(location_file, delimiter=',')
                N_1_4 = location[8:, :]
                N_1_4_flatten = N_1_4.flatten()
                N_1_4_flatten_pad = np.pad(N_1_4_flatten, (0, 3 * (4 - np.size(N_1_4, axis=0))), mode='constant',
                                           constant_values=0) / 4

                map_matrix = np.zeros((12, 12))
                for j in range(np.size(element, axis=0)):
                    map_matrix[element[j, 0], element[j, 1]] = 1
                    map_matrix[element[j, 1], element[j, 0]] = 1
                map_matrix2 = np.triu(map_matrix, k=0)
                N_1_12 = np.zeros(11)
                for j in range(11):
                    num = 0
                    for k in range(12):
                        num = num + map_matrix2[j, k] * 2 ** (11 - k)
                    full = 2 ** (11 - j) - 1
                    num_d = num / full
                    N_1_12[j] = num_d
                N = np.append(N_1_4_flatten_pad, N_1_12)
                data_input[(index - 1) * 1000 + i, :] = N

        x_data = data_input

        np.savetxt(self.path + "x_data.txt", x_data, delimiter=',')
        np.savetxt(self.path + "young_modulus_data.txt", data_young_modulus, delimiter=',')
        np.savetxt(self.path + "yield_stress_data.txt", data_yield_stress, delimiter=',')
