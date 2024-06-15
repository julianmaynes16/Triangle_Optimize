#List of sequences:
# 0 1 2 3
# 0 1 3 2 
# 0 2 3 1
# 0 2 1 3
# 0 3 2 1
# 0 3 1 2
# 1 0 2 3
# 1 0 3 2
# 1 2 3 0
# 1 2 0 3
# 1 3 2 0
# 1 3 0 2 
# 2 0 1 2
# 2 0 2 1
# 2 1 3 0
# 2 1 0 3
# 2 3 0 1
# 2 3 1 0
# 3 0 1 2
# 3 0 2 1
# 3 1 2 0
# 3 1 0 2
# 3 2 0 1
# 3 2 1 0

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import itertools

# make triangles with legs on ground
# stability =

theta = 0.20

triangles =  [
((1,1,1),(2,2,2),(1,3,4)),
((2,3,4),(9,9,9),(3,4,5)),
]


def rotation_matrix(angle):
    cos_ang = np.cos(angle)
    sin_ang = np.sin(angle)
    return_matrix = np.array([[cos_ang, -sin_ang, 0],
                              [sin_ang, cos_ang, 0],
                              [0, 0, 1]])
    return return_matrix
## returns rotated point about an axis
def matrix_multiply(point, angle):
    rot_matrix = rotation_matrix(angle)
    return np.dot(rot_matrix, point)

def pointDistance(point1,point2):
    return np.sqrt(pow((point2[0]-point1[0]),2) + pow((point2[1]-point1[0]),2))

# returns centroid of triangle made by 3 points. point = [x,y,z]
def getCentroid(point1, point2, point3):
    centroidX = (point1[0] + point2[0] + point3[0])/3
    centroidY = (point1[1] + point2[1] + point3[1])/3
    return np.array([centroidX, centroidY, 0])

def longestSide(point1, point2, point3):
    one_two = pointDistance(point1, point2)
    two_three = pointDistance(point2, point3)
    three_one = pointDistance(point3, point1)
    longest_side = max(one_two, two_three, three_one)
    if longest_side == one_two:
        return np.array([point1,point2])
    elif longest_side == two_three:
        return np.array([point2,point3])
    else:
        return np.array([point3, point1])
    
# gets midpoint: side = [[x1,y1,z1],[x2,y2,z2]]
def getMidpoint(side): 
    return np.array([((side[0][0]+side[1][0])/2),((side[0][1]+side[1][1])/2),0])

# 0.2 radians = 0.8 multiplier on interpolation, 0 radians = 1 multiplier 
def heuristicMultiplier(angle):
    return pow((angle-1),2)

def getStability(point1, point2, point3, angle):
    centroid = getCentroid(point1,point2,point3)
    #get midpoint of longest side
    midpoint = getMidpoint(longestSide(point1, point2, point3))
    midpoint_centroid_dist = pointDistance(midpoint, centroid)
    #linear interpolation from midpoint to centroid
    comX = centroid[0] + heuristicMultiplier(angle) * (midpoint[0]-centroid[0])
    comY = centroid[1] + heuristicMultiplier(angle) * (midpoint[1]-centroid[1])
    com = np.array([comX, comY, 0])
    # stability = distance from com to centroid, lower = better
    stability = np.sqrt(pow((com[0] - centroid[0]),2) + pow((com[1]-centroid[1]),2))
    return stability



ax = plt.subplot(projection="3d")

#ax.add_collection(Poly3DCollection(stand))

ax.set_xlim([0,50])
ax.set_ylim([0,50])
ax.set_zlim([0,10])

# Generate permutations of the numbers 0, 1, 2, 3
permutations = list(itertools.permutations([0, 1, 2, 3]))

# Convert permutations to a NumPy array
np_permutations = np.array(permutations)

stab_list = []
#print(np_permutations)
#assuming front right = 0, 
for sequence in np_permutations:
    # 0 = front left, 1 = back left, 2 = front right, 3 = back right
    l0 = [-16.5, 23,0] # 0
    l1 = [-16.5, -23, 0] # 1
    l2 = [16.5, 23,0] # 2
    l3 = [16.5, -23.0,0] # 3
    stab_list.append(sequence)
    for leg in sequence:
        if leg == 0: 
            #get stability for remaining legs
            stab_stat = getStability(l1, l2, l3,0.2)
            #move leg 
            l0 = matrix_multiply(l0, 0.2)
        elif leg == 1:
            stab_stat = getStability(l0, l2, l3,0.2)
            l1 = matrix_multiply(l1, 0.2)
        elif leg == 2: 
           stab_stat = getStability(l0, l1, l3,0.2)
           l2 = matrix_multiply(l2, 0.2)
        else:
            stab_stat = getStability(l0, l1, l2,0.2)
            l3 = matrix_multiply(l3, 0.2)
        stab_list.append(stab_stat)

print (np_permutations)
print(stab_list)
