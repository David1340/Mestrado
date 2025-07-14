# === QGPSO for experiments with obstacles ===

# === Import Python libraries ===
import random
random.seed(42)  # Set seed for reproducibility
from random import random, uniform
import numpy as np

# === Import custom functions ===
from funcoes import distancia, orientacao, deteccao_de_colisao
from funcoes_quanticas import AAQ2

# === Import manipulator functions ===
from manipulador_15dof import *

class particle:
    def __init__(self, position, dimension):
        self.p = position             # Current position (robot configuration)
        self.v = np.zeros(dimension) # Current velocity
        self.n = dimension           # Dimensionality
        self.d = 0                   # Position error (magnitude)
        self.o = np.array([0, 0, 0]) # Orientation error (magnitude)
        self.f = np.Inf              # Current fitness value

    def update_position(self, qbest, L):
        """Update the particle's position and velocity"""
        c1 = 2  # Global best influence
        for i in range(self.n):
            w = 0.5 + random() / 2
            vmax = 0.2  # Velocity cap

            self.v[i] = w * self.v[i] + c1 * random() * (qbest[i] - self.p[i])
            self.v[i] = max(min(self.v[i], vmax), -vmax)

            self.p[i] += self.v[i]
            self.p[i] = max(min(self.p[i], L[i]), -L[i])

    def update_fuction(self, target_position, target_orientation, obstacles):
        """Evaluate fitness considering obstacle collisions and pose error"""
        points, orientation = Cinematica_Direta(self.p, True)
        end = points.shape[1] - 1

        # Collision detection
        for sphere in obstacles:
            for i in range(end):
                r = sphere.get_raio() + getRaio()
                if deteccao_de_colisao(points[:, i], points[:, i + 1], sphere.get_centro(), r):
                    self.f = np.Inf
                    return

        p = points[:, end]
        self.o = distancia(orientacao(orientation), target_orientation, 3)
        self.d = distancia(p, target_position, 3)

        # Cost function
        k1 = 0.1  # Orientation weight
        k2 = 1    # Position weight
        self.f = k1 * self.o + k2 * self.d


def _QGPSO(target_position, target_orientation, num_particles, dimension, L, error_min, Kmax, obstacles):
    """Internal function for executing QGPSO with obstacle handling"""
    counter = 0
    n2 = int(np.log2(num_particles))
    k = Kmax
    swarm = []
    threshold = 1
    fitness_history = []

    # Initialize swarm
    for _ in range(num_particles):
        p = [uniform(-L[i], L[i]) for i in range(dimension)]
        part = particle(p, dimension)
        part.update_fuction(target_position, target_orientation, obstacles)
        swarm.append(part)

    # Initialize global best
    qbest = swarm[0].p.copy()
    f = swarm[0].f
    for p in swarm:
        if p.f < f:
            qbest = p.p.copy()
            f = p.f

    # Main loop
    for j in range(k):
        candidates = []
        for i in range(num_particles):
            swarm[i].update_position(qbest, L)
            swarm[i].update_fuction(target_position, target_orientation, obstacles)

            if swarm[i].f < threshold * f:
                candidates.append(i)

        # Quantum selection step
        idx, count = AAQ2(n2, candidates)
        counter += count

        if swarm[idx].f < threshold * f:
            qbest = swarm[idx].p.copy()
            f = swarm[idx].f
        else:
            if len(candidates) > 0:
                print("AAQ failed")

        fitness_history.append(f)

        if f <= error_min:
            break

    return [f, j + 1, counter]


def QGPSO(pos_target, orient_target, error_min, Kmax, obstacles, N):
    """Wrapper for QGPSO that prepares inputs"""
    orient_target = orientacao(orient_target)
    dimension = getNumberJoints()
    limits = getLimits()

    return _QGPSO(pos_target, orient_target, N, dimension, limits, error_min, Kmax, obstacles)
