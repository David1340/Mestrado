# === FRPSO for experiments without obstacles ===

# === Import Python libraries ===
from random import random, uniform
import random
random.seed(42)  # Set seed for reproducibility

import numpy as np
np.random.seed(42) # Set seed for reproducibility
# === Import custom functions ===
from funcoes import distancia, orientacao

# === Import manipulator functions ===
from pioneer_7dof import *

class particle:
    def __init__(self, position, dimension):
        self.p = position           # Current particle position (robot configuration)
        self.bp = position.copy()   # Best position found by the particle
        self.n = dimension          # Dimensionality of the particle
        self.d = 0                  # Distance error (Euclidean)
        self.o = np.array([0, 0, 0])# Orientation error
        self.f = np.Inf             # Current cost/fitness
        self.bf = self.f            # Best fitness found by the particle

    def update_fuction(self, o, o2):
        """Evaluate the fitness function of the particle"""

        # Check joint limits
        limits = getLimits()
        for (qi, li) in zip(self.p, limits):
            if np.abs(qi) > np.abs(li):
                self.f = np.Inf
                return

        # Compute forward kinematics
        p, orient = Cinematica_Direta3(self.p)

        # Compute orientation error
        self.o = distancia(orientacao(orient), o2, 3)

        # Compute position error
        self.d = distancia(p, o, 3)

        # Compute fitness
        k1 = 0.1  # Weight for orientation error
        k2 = 1    # Weight for position error
        self.f = (k1 * self.o) + (k2 * self.d)

        # Update best fitness and position
        if self.f < self.bf:
            self.bf = self.f
            self.bp = self.p.copy()


def FRPSO2(o, o2, number, n, L, erro_min, Kmax):
    """Run the FRPSO algorithm"""

    k = Kmax  # Maximum number of iterations
    q = []
    Nbests = 5  # Number of best particles to retain
    tau = 0.5   # Spread control parameter

    # === Initialize particles ===
    for i in range(number):
        p = [uniform(-L[i2], L[i2]) for i2 in range(n)]
        q.append(particle(p, n))
        q[i].update_fuction(o, o2)

    # === Initialize top-N best particles ===
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

        # Generate new particles around the best ones
        sig = f / tau
        for N in range(Nbests):
            for i in range(int(number / (Nbests + 1))):
                p = sig * np.random.randn(n)
                for i2 in range(n):
                    p[i2] = qbests[N][i2] + p[i2]
                q.append(particle(p, n))

        # Add random particles
        for i in range(number - len(q)):
            p = [uniform(-L[i2], L[i2]) for i2 in range(n)]
            q.append(particle(p, n))

        # Evaluate particles
        for i in range(number):
            q[i].update_fuction(o, o2)

            if max(qvalues) > q[i].f:
                for i2 in range(Nbests):
                    if qvalues[i2] == max(qvalues):
                        qvalues[i2] = q[i].f
                        qbests[i2] = q[i].p.copy()
                        break
                f = min(qvalues)
                qBest = qbests[np.argmin(qvalues)]

        # Stopping criterion
        if f <= erro_min:
            break

    return [f, j + 1, number * (j + 1)]


def FRPSO(posicaod, orientacaod, erro_min, Kmax,N):
    """Wrapper for FRPSO with input pre-processing"""

    orientacaod = orientacao(orientacaod)
    numero_particulas = N
    dimensao = getNumberJoints()
    L = getLimits()  # Joint limits

    return FRPSO2(posicaod, orientacaod, numero_particulas, dimensao, L, erro_min, Kmax)
