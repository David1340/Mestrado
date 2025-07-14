# === PSO for experiments with obstacles ===

# === Import Python libraries ===
import random
random.seed(42)  # Set seed for reproducibility
from random import random, uniform
import numpy as np

# === Import custom functions ===
from funcoes import distancia, orientacao, deteccao_de_colisao

# === Import manipulator functions ===
from manipulador_15dof import *

class particle:
    def __init__(self, position, dimension):
        self.p = position             # Current particle position (robot configuration)
        self.v = np.zeros(dimension) # Current velocity
        self.bp = position.copy()    # Best position visited by this particle
        self.n = dimension           # Particle dimensionality
        self.d = 0                   # Distance error (Euclidean)
        self.o = np.array([0, 0, 0]) # Orientation error
        self.f = np.Inf              # Current cost/fitness
        self.bf = self.f             # Best fitness found by the particle

    def update_position(self, qbest, L):
        """Update the particle's position based on velocity and bounds"""
        c1 = 1    # Global component coefficient
        c2 = 0.1  # Personal component coefficient

        for i in range(self.n):
            w = 0.5 + random() / 2   # Inertia weight
            vmax = 0.2               # Max velocity

            # Velocity update
            self.v[i] = (
                w * self.v[i] +
                c1 * random() * (qbest[i] - self.p[i]) +
                c2 * random() * (self.bp[i] - self.p[i])
            )

            # Clamp velocity
            self.v[i] = max(min(self.v[i], vmax), -vmax)

            # Position update with bounds
            self.p[i] += self.v[i]
            self.p[i] = max(min(self.p[i], L[i]), -L[i])

    def update_fuction(self, o, o2, esferas):
        """Evaluate the fitness function of the particle considering obstacles"""
        # o: desired position
        # o2: desired orientation

        pontos, orient = Cinematica_Direta(self.p, True)
        end = np.shape(pontos)[1] - 1

        # Collision detection
        for esfera in esferas:
            for i in range(end):
                r = esfera.get_raio() + getRaio()
                if deteccao_de_colisao(pontos[:, i], pontos[:, i + 1], esfera.get_centro(), r):
                    self.f = np.Inf
                    return

        # Compute final end-effector position and errors
        p = pontos[:, end]
        self.o = distancia(orientacao(orient), o2, 3)
        self.d = distancia(p, o, 3)

        # Fitness function (weighted sum)
        k1 = 0.1  # Orientation weight
        k2 = 1    # Position weight
        self.f = (k1 * self.o) + (k2 * self.d)

        # Update best personal position
        if self.f < self.bf:
            self.bf = self.f
            self.bp = self.p.copy()


def PSO2(o, o2, number, n, L, erro_min, Kmax, esferas):
    """Run the Particle Swarm Optimization algorithm with obstacle consideration"""
    k = Kmax
    q = []
    evolucao_qbets = []

    # === Initialize particles ===
    for i in range(number):
        p = [uniform(-L[i2], L[i2]) for i2 in range(n)]
        q.append(particle(p, n))
        q[i].update_fuction(o, o2, esferas)

    # === Initialize global best ===
    qbest = q[0].p.copy()
    f = q[0].f
    for i in range(number):
        if q[i].f < f:
            qbest = q[i].p.copy()
            f = q[i].f

    # === Main PSO loop ===
    for j in range(k):
        for i in range(number):
            q[i].update_position(qbest, L)
            q[i].update_fuction(o, o2, esferas)

            if q[i].f < f:
                qbest = q[i].p.copy()
                f = q[i].f

        evolucao_qbets.append(f)

        # Stopping criterion
        if f <= erro_min:
            break

    return [f, j + 1, number * (j + 1)]


def PSO(posicaod, orientacaod, erro_min, Kmax, esferas, N):
    """Wrapper for PSO with preprocessing of inputs"""
    orientacaod = orientacao(orientacaod)
    numero_particulas = N
    dimensao = getNumberJoints()
    L = getLimits()

    return PSO2(posicaod, orientacaod, numero_particulas, dimensao, L, erro_min, Kmax, esferas)
