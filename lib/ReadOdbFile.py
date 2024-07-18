# -*- coding: utf-8 -*-
# @Time    :2024/6/6 10:58
# @Author  :MENGLong Li
# @File    :ReadOdbFile.py
# @Software:PyCharm

import numpy as np
from abaqus import *
from abaqusConstants import *
import os


class ReadOdbFile:
    def __init__(self, num_group, num_index, path):
        self.num_group = num_group
        self.num_index = num_index
        self.path = path  # ODB文件所在路径
        self.save_path = os.path.abspath(__file__)

    def run(self):
        for index in range(1, self.num_group + 1):

            os.chdir(self.path)
            U = []
            F = []
            for i in range(self.num_index):
                name = self.path + "\\Job-BJX-%s-No%s.odb" % (i, index)
                odb = session.openOdb(name=name)
                step = odb.steps['Step-1']
                n_frames = np.size(step.frames,0)
                for frame_i in range(n_frames):
                    dis_field = step.frames[frame_i].fieldOutputs['U']
                    node_set = odb.rootAssembly.nodeSets["ASSEMBLY_CONSTRAINT-1_REFERENCE_POINT"]
                    local_disp = dis_field.getSubset(region=node_set)
                    U.append(local_disp.values[0].data[1])
                    ###
                    F_field = step.frames[frame_i].fieldOutputs['RF']
                    local_filed = F_field.getSubset(region=node_set)
                    F.append(local_filed.values[0].data[1])
                ###
                str_data = []
                for i in range(n_frames):
                    s1 = "%.5f, %.5f" % (U[i], F[i])
                    str_data.append(s1)
                with open(self.save_path + "\\data\\Data-BJX-%s-No%s.txt" % (i, index), 'w') as f:
                    for data in str_data:
                        f.write('%s,\n' % data)
                f.close()
