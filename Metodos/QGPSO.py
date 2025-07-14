# === QGPSO for experiments without obstacles ===

# === Import Python libraries ===
import random
random.seed(42)  # Set seed for reproducibility
from random import random, uniform
import numpy as np

# === Import custom functions ===
from funcoes import distancia, orientacao
from funcoes_quanticas import AAQ2

# === Import manipulator functions ===
from pioneer_7dof import *

class particle:
    def __init__(self, position, dimension):
        self.p = position             # Current particle position (robot configuration)
        self.v = np.zeros(dimension) # Current velocity
        self.n = dimension           # Particle dimensionality
        self.d = 0                   # Positional error (Euclidean)
        self.o = np.array([0, 0, 0]) # Orientation error
        self.f = np.Inf              # Current cost/fitness

    def update_position(self, qbest, L):
        """Update the particle's position and velocity"""
        c1 = 2  # Group influence coefficient

        for i in range(self.n):
            w = 0.5 + random() / 2
            vmax = 0.2  # Maximum velocity

            # Velocity update
            self.v[i] = w * self.v[i] + c1 * random() * (qbest[i] - self.p[i])

            # Velocity clamping
            self.v[i] = max(min(self.v[i], vmax), -vmax)

            # Position update with bounds check
            self.p[i] += self.v[i]
            self.p[i] = max(min(self.p[i], L[i]), -L[i])

    def update_fuction(self, o, o2):
        """Evaluate the fitness function of the particle"""
        # o: desired position
        # o2: desired orientation

        # Compute forward kinematics
        p, orient = Cinematica_Direta3(self.p)

        # Compute orientation and position errors
        self.o = distancia(orientacao(orient), o2, 3)
        self.d = distancia(p, o, 3)

        # Compute fitness value
        k1 = 0.1  # Weight for orientation error
        k2 = 1    # Weight for position error
        self.f = (k1 * self.o) + (k2 * self.d)


def QGPSO2(o, o2, number, n, L, erro_min, Kmax):
    """Run the QGPSO algorithm"""

    contador = 0  # Quantum evaluation counter
    n2 = np.log2(number).astype(int)
    k = Kmax
    c = 1  # Threshold coefficient for selecting better particles
    q = []

    # === Initialize particles ===
    for i in range(number):
        p = [uniform(-L[i2], L[i2]) for i2 in range(n)]
        q.append(particle(p, n))
        q[i].update_fuction(o, o2)

    # === Initialize global best ===
    qbest = q[0].p.copy()
    f = q[0].f
    for i in range(number):
        if q[i].f < f:
            qbest = q[i].p.copy()
            f = q[i].f

    # === Main loop ===
    for j in range(k):
        indices = []

        for i in range(number):
            q[i].update_position(qbest, L)
            q[i].update_fuction(o, o2)

            # Select candidates that are significantly better than current best
            if q[i].f < c * f:
                indices.append(i)

        # Quantum Grover-inspired selection (AAQ2)
        [i, cont] = AAQ2(n2, indices)
        contador += cont

        if q[i].f < c * f:
            qbest = q[i].p.copy()
            f = q[i].f
        else:
            if len(indices) > 0:
                print("AAQ failed")

        # Stopping criterion
        if f <= erro_min:
            break

    return [f, j + 1, contador]


def QGPSO(posicaod, orientacaod, erro_min, Kmax,N):
    """Wrapper for QGPSO with input pre-processing"""

    orientacaod = orientacao(orientacaod)
    numero_particulas = N
    dimensao = getNumberJoints()  # Robot dimensionality
    L = getLimits()               # Joint limits

    return QGPSO2(posicaod, orientacaod, numero_particulas, dimensao, L, erro_min, Kmax)
