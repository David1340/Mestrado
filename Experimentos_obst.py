#Autor David Oliveira.
#Mestrando em Engenharia Elétrica da Universidade Federal de Campina Grande-UFCG.
#Membro do Grupo de Pesquisa em Robotica da UFS-GPRUFS.
#Implementação do Fully Resampled Particle Swarm Optimizarion

#Import das bibliotecas
import numpy as np
from random import uniform
import sys
import os
import platform
from funcoes import Esfera

#Import diretorio
diretorio_atual = os.getcwd()
sys.path.append(diretorio_atual)
if(platform.system == 'Windows'):
    sys.path.append(diretorio_atual + '\Metodos_com_obst')
else:
    sys.path.append(diretorio_atual + '/Metodos_com_obst')

#Import dos métodos
from FRPSO import FRPSO
from pioneer_7dof import *

#Configurações do experimento
Kmax = 1000
erro_min = 0.001
repeticoes = 1000

#parâmetros do manipulador
qlim = getLimits() 
n = getNumberJoints()
q = np.zeros([n,1])

kFRPSO = []
evolucao_qbets_list = np.zeros([repeticoes,Kmax])
tc = [0]
mi = tc.copy()

#obstáculos
esferas = []
a = 0.1
r = 0.025
h1 = 1*0.3
h2 = 1*0.2
h3 = 1*0.1
esferas.append(Esfera(a,a,h1,r))
esferas.append(Esfera(-a,-a,h1,r))
esferas.append(Esfera(-a,a,h1,r))   
esferas.append(Esfera(a,-a,h1,r))

esferas.append(Esfera(a,a,h3,r))
esferas.append(Esfera(-a,-a,h3,r))
esferas.append(Esfera(-a,a,h3,r))
esferas.append(Esfera(a,-a,h3,r))


esferas.append(Esfera(a,0,h2,r))
esferas.append(Esfera(-a,0,h2,r))
esferas.append(Esfera(0,a,h2,r))
esferas.append(Esfera(0,-a,h2,r))

esferas = []

for i in range(repeticoes):
    print('i:',i)

    #Gerando a configuração inicial
    for i2 in range(np.size(q)):
        q[i2] = uniform(-qlim[i2],qlim[i2])

    [posicaod,orientacaod] = random_pose(esferas)
    
    #[erro,k,evolucao_qbets] = PSO(posicaod,orientacaod,erro_min,Kmax,esferas)
    [erro,k,evolucao_qbets] = FRPSO(posicaod,orientacaod,erro_min,Kmax,esferas)
    evolucao_qbets_list[i,0:len(evolucao_qbets)] = evolucao_qbets
    evolucao_qbets_list[i,len(evolucao_qbets)-1:Kmax] = evolucao_qbets[-1]
    if(erro < erro_min):
        kFRPSO.append(k)
        print(k)


tc[0] = (len(kFRPSO)/repeticoes) * 100

mi[0] = np.mean(kFRPSO)

print(tc)
print(np.round(mi,2))
metodos = ["FRPSO"]

if(len(esferas) == 0):
    arquivo = open("Experimentos_com_obst.txt", "w")
else:
    rquivo = open("Experimentos_sem_obst.txt", "w")

arquivo.write("Metodos: " + str(metodos) + "\n")
arquivo.write("tc: " + str(np.round(tc,2)) + "\n")
arquivo.write("mi: " + str(np.round(mi,2))+ "\n")
arquivo.write("Kmax: " + str(Kmax))
arquivo.close()

evolucao_qbets_list = np.mean(evolucao_qbets_list,0)
evolucao_qbets_list = np.round(evolucao_qbets_list,4)
arquivo = open("Evolucao_fbest.txt", "w")
arquivo.write(str(evolucao_qbets_list))
arquivo.close()