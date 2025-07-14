# Experiments with obstacles

# === Library imports ===
import numpy as np
import random
random.seed(42)  # Set seed for reproducibility
import sys
import os
import platform
from funcoes import Esfera

# === Add method directory to path ===
current_dir = os.getcwd()
sys.path.append(current_dir)
if platform.system() == 'Windows':
    sys.path.append(current_dir + '\\Metodos_com_obst')
else:
    sys.path.append(current_dir + '/Metodos_com_obst')

# === Import optimization methods ===
from FRPSO import MFRPSO
from PSO import PSO
from QGPSO import QGPSO

# === Import robotic manipulator ===
from manipulador_15dof import * #this must also be put in the methods files

# === Experiment configuration ===
methods = ["PSO", "QGPSO", "FRPSO"]
N = 256 #size of the swarms
Kmax = 300
erro_min = 0.01 * (sum(getLengthElos()) + 0.075)
repetitions = 100

# === Manipulator parameters ===
qlim = getLimits()
n = getNumberJoints()
q = np.zeros([n, 1])

# === PSO results containers ===
klist = []
kcont = []
tc = [0]
mi = tc.copy()
mf = tc.copy()

# === QGPSO results containers ===
klist2 = []
kcont2 = []
tc2 = [0]
mi2 = tc2.copy()
mf2 = tc2.copy()

# === MFRPSO results containers ===
klist3 = []
kcont3 = []
tc3 = [0]
mi3 = tc3.copy()
mf3 = tc3.copy()


# === Define obstacle spheres ===
spheres = []
a = 0.1
r = 0.025
h1 = 2 * 0.3
h2 = 2 * 0.2
h3 = 2 * 0.1

# Top level
spheres.extend([
    Esfera(a, a, h1, r),
    Esfera(-a, -a, h1, r),
    Esfera(-a, a, h1, r),
    Esfera(a, -a, h1, r)
])

# Bottom level
spheres.extend([
    Esfera(a, a, h3, r),
    Esfera(-a, -a, h3, r),
    Esfera(-a, a, h3, r),
    Esfera(a, -a, h3, r)
])

# Middle level
spheres.extend([
    Esfera(a, 0, h2, r),
    Esfera(-a, 0, h2, r),
    Esfera(0, a, h2, r),
    Esfera(0, -a, h2, r)
])

# === Run the experiments ===
for i in range(repetitions):
    print('i:', i)

    # Generate random goal pose (collision-free)
    posicaod, orientacaod = random_pose(spheres)

    # Run methods with obstacles
    error, k, count = PSO(posicaod, orientacaod, erro_min, Kmax, spheres, N)
    error2, k2, count2 = QGPSO(posicaod, orientacaod, erro_min, Kmax, spheres, N)
    error3, k3, count3 = MFRPSO(posicaod, orientacaod, erro_min, Kmax, spheres, N)

    if error < erro_min:
        klist.append(k)
        kcont.append(count)

    if error2 < erro_min:
        klist2.append(k2)
        kcont2.append(count2)

    if error3 < erro_min:
        klist3.append(k3)
        kcont3.append(count3)

# === PSO statistics ===
print("PSO")
tc[0] = (len(klist) / repetitions) * 100
mi[0] = np.mean(klist)
print(np.std(klist))
mf[0] = np.mean(kcont)

print(tc)
print(np.round(mi, 2))
print(np.round(mf, 2))

# === QGPSO statistics ===
print("QGPSO")
tc2[0] = (len(klist2) / repetitions) * 100
mi2[0] = np.mean(klist2)
mf2[0] = np.mean(kcont2)

print(tc2)
print(np.round(mi2, 2))
print(np.round(mf2, 2))

# === FRPSO statistics ===
print("MFRPSO")
tc3[0] = (len(klist3) / repetitions) * 100
mi3[0] = np.mean(klist3)
mf3[0] = np.mean(kcont3)

print(tc3)
print(np.round(mi3, 2))
print(np.round(mf3, 2))

# === Save results to file ===
with open("experiments_wit_obst.txt", "w") as file:
    file.write("Methods: " + str(methods) + "\n")
    file.write("tc: " + str(np.round(tc, 2)) + ", " + str(np.round(tc2, 2)) + ", " + str(np.round(tc3, 2)) + "\n")
    file.write("mi: " + str(np.round(mi, 2)) + ", " + str(np.round(mi2, 2)) + ", " + str(np.round(mi3, 2)) + "\n")
    file.write("mf: " + str(mf) + ", " + str(mf2) + ", " + str(mf3) + "\n")
    file.write("Kmax: " + str(Kmax))
