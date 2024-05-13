#Autor David Oliveira.
#Mestrando em Engenharia Elétrica da Universidade Federal de Campina Grande-UFCG.
#Membro do Grupo de Pesquisa em Robotica da UFS-GPRUFS.

import numpy as np
from random import random

def oracle(n,indices):
    O = np.eye(2**n)
    for i in indices:
        O[i,i] = -1
    return O

def Hadamard(n):
    H1 = np.array([[1,1],[1,-1]]) * (1/np.sqrt(2))
        
    if(n > 1):
        H = 1
        for _ in range(n):
            H = np.kron(H,H1)
        
        return H
    else:
        return H1
        
def AAQ(n,M,indices):
    
    X = np.array([[0,1],[1,0]]) #Operador Inversor
    q = np.array([[1,0]]).T #Qubit Oracle
    x = q.copy() #Initial State
    u = np.ones([2**n,1])
    u = u/np.linalg.norm(u) #uniform state
    N = 2**n
    O = oracle(n,indices)

    for _ in range(1,n):
        x = np.kron(x,q)

    H = Hadamard(1)
    H_n = Hadamard(n)
    q = H@X@q
    x = H_n@x
    I = np.eye(2**n)
    G = (2*u*u.T - I)@O

    if(M < 1):
        return x

    R = (np.floor(np.pi*np.sqrt(N/M)/4)).astype(int)

    for _ in range(R):
        x = G@x
        
    return x

def medir(x):
    P = np.cumsum(np.abs(x)**2)
    r = random()
    for i in range(len(P)):
        if(r < P[i]):
            return i

    

def AAQ2(n,indices):
    contador = 0
    #Amplificação de amplitude quântica quando o M é desconhecido
    X = np.array([[0,1],[1,0]]) #Operador Inversor
    q = np.array([[1,0]]).T #Qubit Oracle
    x = q.copy() #Initial State
    u = np.ones([2**n,1])
    u = u/np.linalg.norm(u) #uniform state
    N = 2**n
    O = oracle(n,indices)

    for _ in range(1,n):
        x = np.kron(x,q)

    H = Hadamard(1)
    H_n = Hadamard(n)
    q = H@X@q
    x = H_n@x
    I = np.eye(2**n)
    G = (2*u*u.T - I)@O

    x0 = x.copy()
    m = 1
    lamb = 6/5
    while(True):
        j = (np.round((m-1)*random())).astype(int)
        x = x0.copy()
        for _ in range(j):
            contador  = contador + 1
            x = G@x
        
        x_medido = medir(x)
        contador  = contador + 1 #checagem clássica
        if(any(i == x_medido for i in indices)):         
            return [x_medido, contador]
        elif(m >= np.sqrt(N)):
            return [medir(x0), contador]
        else:
            m = np.min([lamb*m,np.sqrt(N)])