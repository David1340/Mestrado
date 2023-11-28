#Autor David Oliveira.
#Estudante de Engenharia Eletrônica da Universidade Federal de Sergipe-UFS.
#Membro do Grupo de Pesquisa em Robotica da UFS-GPRUFS.
#Experimentos apenas a posição .
#Implementações feitas durante durante a iniciação científica intitulada:
#PIB10456-2021 - Soluções de cinemática inversa de robôs manipuladores seriais com restrições físicas
#Durante o período: PIBIC 2021/2022 (01/09/2021 a 31/08/2022).

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
    sys.path.append(diretorio_atual + '\Metodos2')
else:
    sys.path.append(diretorio_atual + '/Metodos2')

#Import dos métodos
from PSO import PSO
from FRPSO import FRPSO
from manipulador_15dof import *

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
arquivo = open("ExperimentoB3.txt", "w")
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