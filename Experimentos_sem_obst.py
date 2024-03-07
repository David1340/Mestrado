#Autor David Oliveira.
#Mestrando em Engenharia Elétrica da Universidade Federal de Campina Grande-UFCG.
#Membro do Grupo de Pesquisa em Robotica da UFS-GPRUFS.
#Implementação do Fully Resampled Particle Swarm Optimizarion

#Import das bibliotecas
import numpy as np
import random
from random import uniform
import sys
import os
import platform

#Import diretorio
diretorio_atual = os.getcwd()
sys.path.append(diretorio_atual)
if(platform.system == 'Windows'):
    sys.path.append(diretorio_atual + '\Metodos')
else:
    sys.path.append(diretorio_atual + '/Metodos')

#Import dos métodos
from Novo_PSO_Quantico import PSO
from pioneer_7dof import *
#Configurações do experimento
Kmax = 300
erro_min = 0.01*(sum(getLengthElos()) + 0.075)  #0.001
print(erro_min)
repeticoes = 1000

#parâmetros do manipulador
qlim = getLimits() 
n = getNumberJoints()
q = np.zeros([n,1])

klist = []

tc = [0]
mi = tc.copy()

for i in range(repeticoes):
    print('i:',i)

    #Gerando a configuração inicial
    for i2 in range(np.size(q)):
        q[i2] = uniform(-qlim[i2],qlim[i2])

    [posicaod,orientacaod] = random_pose()
    
    [erro,k] = PSO(posicaod,orientacaod,erro_min,Kmax)

    if(erro < erro_min):
        klist.append(k)


tc[0] = (len(klist)/repeticoes) * 100

mi[0] = np.mean(klist)

print(tc)
print(np.round(mi,2))
metodos = ["Novo_PSO"]
arquivo = open("Experimentos_sem_obst.txt", "w")
arquivo.write("Metodos: " + str(metodos) + "\n")
arquivo.write("tc: " + str(np.round(tc,2)) + "\n")
arquivo.write("mi: " + str(np.round(mi,2))+ "\n")
arquivo.write("Kmax: " + str(Kmax))
arquivo.close()
