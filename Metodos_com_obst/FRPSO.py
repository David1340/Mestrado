# === FRPSO for experiments with obstacles ===

# === Import Python libraries ===
import random
random.seed(42)  # Set seed for reproducibility
from random import random, uniform
import numpy as np
np.random.seed(42) # Set seed for reproducibility

# === Import custom functions ===
from funcoes import deteccao_de_colisao, distancia, orientacao

# === Import manipulator functions ===
from manipulador_15dof import *

class particle:
    def __init__(self, position, dimension):
        self.p = position           # Current particle position (robot configuration)
        self.n = dimension          # Particle dimensionality
        self.d = 0                  # Distance error (Euclidean)
        self.o = np.array([0, 0, 0])# Orientation error
        self.f = np.Inf             # Fitness function (cost)

    def update_fuction(self, position, orientation, esferas):
        """Evaluate the fitness of the particle considering collisions"""
        limits = getLimits()

        # Check joint limits
        for (qi, li) in zip(self.p, limits):
            if np.abs(qi) > np.abs(li):
                self.f = np.Inf
                return

        # Get forward kinematics and robot links
        pontos, orient = Cinematica_Direta(self.p, True)
        end = np.shape(pontos)[1] - 1

        # Check for collisions with obstacles (spheres)
        for esfera in esferas:
            for i in range(end):
                r = esfera.get_raio() + getRaio()
                if deteccao_de_colisao(pontos[:, i], pontos[:, i+1], esfera.get_centro(), r):
                    self.f = np.Inf
                    return

        # Compute final position and orientation error
        p = pontos[:, end]
        self.o = distancia(orientacao(orient), orientation, 3)
        self.d = distancia(p, position, 3)

        # Compute fitness function (weighted sum of errors)
        k1 = 0.1  # Orientation weight
        k2 = 1    # Position weight
        self.f = (k1 * self.o) + (k2 * self.d)


def _MFRPSO(position, orientation, number, n, L, erro_min, Kmax, esferas):
    """Run the Modified FRPSO algorithm with obstacle avoidance"""
    k = Kmax
    q = []
    Nbests = 5  # Number of elite particles
    tau = 0.5   # Spread factor
    evolucao_qbets = []

    # === Initialize particles ===
    for i in range(number):
        p = [uniform(-L[i2], L[i2]) for i2 in range(n)]
        q.append(particle(p, n))
        q[i].update_fuction(position, orientation, esferas)

    # === Initialize elite particles ===
    qbests = []
    qvalues = []
    f = np.inf
    for i in range(Nbests):
        qbests.append(q[i].p.copy())
        qvalues.append(q[i].f)
        f = max(f, q[i].f)

    for i in range(number):
        if max(qvalues) > q[i].f:
            for j in range(Nbests):
                if qvalues[j] == max(qvalues):
                    qvalues[j] = q[i].f
                    qbests[j] = q[i].p.copy()
                    break
            f = min(qvalues)

    # === Main FRPSO loop ===
    for j in range(k):
        q = []

        sig = f / tau
        for N in range(Nbests):
            for i in range(int(number / (Nbests + 1))):
                p = sig * np.random.randn(n)
                for i2 in range(n):
                    p[i2] = qbests[N][i2] + p[i2]
                q.append(particle(p, n))

        # Remaining particles are random
        for i in range(number - len(q)):
            p = [uniform(-L[i2], L[i2]) for i2 in range(n)]
            q.append(particle(p, n))

        # Evaluate fitness and update elite
        for i in range(number):
            q[i].update_fuction(position, orientation, esferas)
            if max(qvalues) > q[i].f:
                for i2 in range(Nbests):
                    if qvalues[i2] == max(qvalues):
                        qvalues[i2] = q[i].f
                        qbests[i2] = q[i].p.copy()
                        break
                f = min(qvalues)
                qBest = qbests[np.argmin(qvalues)]

        evolucao_qbets.append(f)

        # Stopping criterion
        if f <= erro_min:
            break

    return [f, j + 1, number * (j + 1)]


def MFRPSO(posicaod, orientacaod, erro_min, Kmax, esferas, N):
    """Wrapper function for MFRPSO"""
    orientacaod = orientacao(orientacaod)
    numero_particulas = N
    dimensao = getNumberJoints()
    L = getLimits()

    return _MFRPSO(posicaod, orientacaod, numero_particulas, dimensao, L, erro_min, Kmax, esferas)
