# === PSO for experiments without obstacles ===

# === Import Python libraries ===
import random
random.seed(42)  # Set seed for reproducibility
from random import random, uniform
import numpy as np

# === Import custom functions ===
from funcoes import distancia, orientacao

# === Import manipulator functions ===
from pioneer_7dof import *

class particle:
    def __init__(self, position, dimension):
        self.p = position             # Current particle position (robot configuration)
        self.v = np.zeros(dimension) # Current velocity
        self.bp = position.copy()    # Best position visited by the particle
        self.n = dimension           # Dimensionality of the problem
        self.d = 0                   # Positional error (Euclidean distance)
        self.o = np.array([0, 0, 0]) # Orientation error
        self.f = np.Inf              # Current cost/fitness
        self.bf = self.f             # Best fitness found by this particle

    def update_position(self, qbest, L):
        """Update the particle's position based on velocity and limits"""
        c1 = 1    # Social component
        c2 = 0.1  # Cognitive component

        for i in range(self.n):
            w = 0.5 + random() / 2  # Inertia weight
            vmax = 0.2

            self.v[i] = (w * self.v[i] +
                         c1 * random() * (qbest[i] - self.p[i]) +
                         c2 * random() * (self.bp[i] - self.p[i]))

            # Velocity clamping
            self.v[i] = max(min(self.v[i], vmax), -vmax)

            # Position update with boundary check
            self.p[i] += self.v[i]
            self.p[i] = max(min(self.p[i], L[i]), -L[i])

    def update_fuction(self, o, o2):
        """Evaluate the fitness function of the particle"""
        # o: desired position
        # o2: desired orientation

        # Compute forward kinematics
        p, orient = Cinematica_Direta3(self.p)

        # Orientation error (magnitude)
        self.o = distancia(orientacao(orient), o2, 3)

        # Positional error (Euclidean distance)
        self.d = distancia(p, o, 3)

        # Fitness function (weighted sum of errors)
        k1 = 0.1  # Orientation weight
        k2 = 1    # Position weight
        self.f = (k1 * self.o) + (k2 * self.d)

        # Update best fitness and position
        if self.f < self.bf:
            self.bf = self.f
            self.bp = self.p.copy()


def PSO2(o, o2, number, n, L, erro_min, Kmax):
    """Run the PSO algorithm given desired pose and parameters"""
    k = Kmax  # Max number of iterations
    q = []

    # Create and evaluate particles
    for i in range(number):
        p = [uniform(-L[i2], L[i2]) for i2 in range(n)]
        q.append(particle(p, n))
        q[i].update_fuction(o, o2)

    # Initialize global best
    qbest = q[0].p.copy()
    f = q[0].f
    for i in range(number):
        if q[i].f < f:
            qbest = q[i].p.copy()
            f = q[i].f

    # Main PSO loop
    for j in range(k):
        for i in range(number):
            q[i].update_position(qbest, L)
            q[i].update_fuction(o, o2)

            # Update global best if necessary
            if q[i].f < f:
                qbest = q[i].p.copy()
                f = q[i].f

        # Stop if error threshold is reached
        if f <= erro_min:
            break

    return [f, j + 1, number * (j + 1)]


def PSO(posicaod, orientacaod, erro_min, Kmax,N):
    """Wrapper function for PSO that prepares the input and runs the optimization"""
    orientacaod = orientacao(orientacaod)
    numero_particulas = N
    dimensao = getNumberJoints()
    L = getLimits()  # Joint limits
    return PSO2(posicaod, orientacaod, numero_particulas, dimensao, L, erro_min, Kmax)
