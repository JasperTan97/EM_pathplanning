import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from helper_classes.vehicles import EdyMobile
import matplotlib.pyplot as plt
import numpy as np

test_edy = EdyMobile(start_position=[0.5, 0.5, (np.pi* 2) * (3/5)])

##########
# Test 1 #
##########
test_edy.visualise()

##########
# Test 2 #
##########
dt = 0.01
L = 0.2
R = 0.2
stage_A = True # move straight
stage_B = False # going rightwards faster
stage_C = False # going rightwards slower
stage_D = False # slowing down and reverse 
X = []
Y = []
Theta = []
V = []
W = []
for _ in range(500):
    test_edy.step(dt, L, R)
    X.append(test_edy._x)
    Y.append(test_edy._y)
    Theta.append(test_edy._theta)
    V.append(test_edy._linear_velocity)
    W.append(test_edy._angular_velocity)
    if test_edy._L_wheel_speed > 0.3 and stage_A:
        stage_A = False
        stage_B = True
        L = 0
    elif test_edy._R_wheel_speed >= 0.5 and stage_B:
        stage_B = False
        stage_C = True
        R = 0
        L = 0.2
    elif test_edy._L_wheel_speed >= 0.5 and stage_C:
        stage_C = False
        stage_D = True
        R = -0.3
        L = -0.3

# plt.subplot(4,1,1)
# plt.plot(X, Y, marker='o', linestyle='-', color='b')
# plt.subplot(4,1,2)
# plt.plot(V, marker='o', linestyle='-', color='b')
# plt.subplot(4,1,3)
# plt.plot(W, marker='o', linestyle='-', color='b')
# plt.subplot(4,1,4)
# plt.plot(Theta, marker='o', linestyle='-', color='b')
# plt.tight_layout()
plt.plot(X, Y, marker='o', linestyle='-', color='b')
# plt.show()
# test_edy.visualise()