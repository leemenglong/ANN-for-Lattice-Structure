# -*- coding: utf-8 -*-
# @Time    :2024/6/17 17:17
# @Author  :MENGLong Li
# @File    :ABAQUSForResult.py
# @Software:PyCharm

import numpy as np
#from abaqus import *
#from abaqusConstants import *
from scipy.spatial import KDTree
import os

pop = 4  # 内部节点个数为1-4个
n = 4  # 每个节点选择最近的4-1个节点作为杆
LengthX = 4  # 点阵结构X边长为4
LengthY = 4  # 点阵结构Y边长为4
LengthZ = 4  # 点阵结构Z边长为4
dia_R = 0.412  # 点阵结构直径为0.412

population = pop + 8  # 内部节点数+八个顶点的个数
Location = np.zeros((population, 3))
Location[0, :] = [0, 0, 0]
Location[1, :] = [LengthX, 0, 0]
Location[2, :] = [0, LengthY, 0]
Location[3, :] = [LengthX, LengthY, 0]
Location[4, :] = [0, 0, LengthZ]
Location[5, :] = [LengthX, 0, LengthZ]
Location[6, :] = [0, LengthY, LengthZ]
Location[7, :] = [LengthX, LengthY, LengthZ]
Location[8, :] =  [0.5*LengthX, 0.5*LengthY,0* LengthZ]
Location[9, :] =  [0.5*LengthX, 0.5*LengthY,1* LengthZ]
Location[10, :] =  [1*LengthX, 0.5*LengthY,0.5* LengthZ]
Location[11, :] =  [0*LengthX, 0.5*LengthY,0.5* LengthZ]
# 计算每个点的最近邻点，并生成element
KD_tree = KDTree(Location)
t, element = KD_tree.query(Location, k=np.linspace(1, n + 1, n + 1, dtype='int'))

# 删除重复的element
paixu = np.zeros((n * population, 2))
ni = 0
for i in range(population):
    for j in range(n):
        paixu[ni, 0] = element[i, 0]
        paixu[ni, 1] = element[i, j + 1]
        if paixu[ni, 0] > paixu[ni, 1]:
            mid = paixu[ni, 0]
            paixu[ni, 0] = paixu[ni, 1]
            paixu[ni, 1] = mid
        ni = ni + 1

element = np.unique(paixu, axis=0)









#----------------------------------
data_input = np.zeros((1, 23))
location = Location
N_1_4 = location[8:, :]
N_1_4_flatten = N_1_4.flatten()
N_1_4_flatten_pad = np.pad(N_1_4_flatten, (0, 3 * (4 - np.size(N_1_4, axis=0))), mode='constant',
                           constant_values=0) / 4

map_matrix = np.zeros((12, 12))
for j in range(np.size(element, axis=0)):
    map_matrix[int(element[j, 0]), int(element[j, 1])] = 1
    map_matrix[int(element[j, 1]), int(element[j, 0])] = 1


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
data_input[0, :] = N















# 开始在ABAQUS根据element和Location生成结构
part1 = mdb.models['Model-1'].Part(name='Part-1', dimensionality=THREE_D, type=DEFORMABLE_BODY)
part1.ReferencePoint(point=(0.0, 0.0, 0.0))
for i in range(population):
    part1.DatumPointByCoordinate(coords=(Location[i, :]))

d = part1.datums
for i in range(np.size(element, axis=0)):
    part1.WirePolyLine(points=((d[int(2 + element[i, 0])], d[int(2 + element[i, 1])])), mergeType=IMPRINT,
                       meshable=ON)

# assembly
assembly = mdb.models['Model-1'].rootAssembly
assembly.DatumCsysByDefault(CARTESIAN)
assembly.Instance(name='Part-1-1', part=part1, dependent=ON)

# Materials
mdb.models['Model-1'].Material(name='TC4')
mdb.models['Model-1'].materials['TC4'].Density(table=((4.51e-09,),))
mdb.models['Model-1'].materials['TC4'].Elastic(table=((110000.0, 0.3),))
mdb.models['Model-1'].materials['TC4'].Plastic(hardening=JOHNSON_COOK, scaleStress=None,
                                               table=((631.2721, 2942.2, 1.1438, 1.1869, 1640.0, 20.0),))
mdb.models['Model-1'].materials['TC4'].plastic.RateDependent(type=JOHNSON_COOK, table=((0.1486, 1.0),))
mdb.models['Model-1'].CircularProfile(name='Profile-1', r=dia_R)
mdb.models['Model-1'].BeamSection(name='Section-1', integration=DURING_ANALYSIS, poissonRatio=0.0,
                                  profile='Profile-1',
                                  material='TC4', temperatureVar=LINEAR, beamSectionOffset=(0.0, 0.0),
                                  consistentMassMatrix=False)

e = part1.edges
LengthEdge = np.size(e)
edges = e[0:LengthEdge]
region = part1.Set(edges=edges, name='Set-1')
part1.SectionAssignment(region=region, sectionName='Section-1', offset=0.0, offsetType=MIDDLE_SURFACE,
                        offsetField='', thicknessAssignment=FROM_SECTION)

# Step
mdb.models['Model-1'].StaticStep(name='Step-1', previous='Initial',
                                 initialInc=0.2, maxInc=0.2)

# interaction
a = mdb.models['Model-1'].rootAssembly
v1 = a.instances['Part-1-1'].vertices
verts1 = v1.findAt(((4.0, 0.0, 4.0),), ((0.0, 0.0, 4.0),), ((4.0,
                                                             0.0, 0.0),))
r1 = a.instances['Part-1-1'].referencePoints
refPoints1 = (r1[1],)
region = a.Set(vertices=verts1, referencePoints=refPoints1, name='Set-2')
mdb.models['Model-1'].EncastreBC(name='BC-1', createStepName='Initial',
                                 region=region, localCsys=None)

a.ReferencePoint(point=(2.0, 4.0, 2.0))
v1 = a.instances['Part-1-1'].vertices
verts1 = v1.findAt(((4.0, 4.0, 4.0),), ((0.0, 4.0, 4.0),), ((4.0, 4.0, 0.0),), ((0.0, 4.0, 0.0),))
r1 = a.referencePoints
refPoints1 = (r1[5],)
region1 = a.Set(referencePoints=refPoints1, name='m_Set-3')
region2 = a.Set(vertices=verts1, name='s_Set-3')
mdb.models['Model-1'].Coupling(name='Constraint-1', controlPoint=region1,
                               surface=region2, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC,
                               alpha=0.0, localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
r1 = a.referencePoints
refPoints1 = (r1[5],)
region = a.Set(referencePoints=refPoints1, name='Set-5')
mdb.models['Model-1'].DisplacementBC(name='BC-2', createStepName='Initial',
                                     region=region, u1=SET, u2=SET, u3=SET, ur1=SET, ur2=SET, ur3=SET,
                                     amplitude=UNSET, distributionType=UNIFORM, fieldName='', localCsys=None)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Step-1')
mdb.models['Model-1'].boundaryConditions['BC-2'].setValuesInStep(
    stepName='Step-1', u2=-0.16)

# Mesh
part1.seedPart(size=0.2, deviationFactor=0.08, minSizeFactor=0.08)
part1.generateMesh()

# assign beam
for i in range(np.size(element, axis=0)):
    e = part1.edges
    node1 = Location[int(element[i, 0]), :]
    node2 = Location[int(element[i, 1]), :]
    find_node = 0.25 * (node2 - node1) + node1
    shiliang = node2 - node1
    if shiliang[0] != 0:
        jieguo = np.array([-shiliang[2] / shiliang[0], 0, 1])
    elif shiliang[1] != 0:
        jieguo = np.array([1, -shiliang[0] / shiliang[1], 0])
    else:
        jieguo = np.array([1, 0, -shiliang[0] / shiliang[2]])
    edges = e.findAt(((find_node[0], find_node[1], find_node[2]),))
    region = part1.Set(edges=edges, name='assign_find_%s' % i)
    part1.assignBeamSectionOrientation(region=region, method=N1_COSINES, n1=(jieguo[0], jieguo[1], jieguo[2]))

part1.regenerate()
assembly.regenerate()



