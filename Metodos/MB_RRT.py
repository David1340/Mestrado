#Autor David Oliveira.
#Estudante de Engenharia Eletrônica da Universidade Federal de Sergipe-UFS.
#Membro do Grupo de Pesquisa em Robotica da UFS-GPRUFS.
#Implementação do Forward And Backward Reaching Inverse Kinematic 
#para encontrar uma configuação q dada uma posição (x,y,z) no espaço para o Pioneer 7DOF
#Implementações feitas durante durante a iniciação científica intitulada:
#PIB10456-2021 - Soluções de cinemática inversa de robôs manipuladores seriais com restrições físicas
#Durante o período: PIBIC 2021/2022 (01/09/2021 a 31/08/2022).

#Import dos módulos do python
from math import pi,acos
import numpy as np

#Import das minhas funções
from funcoes import norm, distancia, S,rotationar_vetor, projecao_ponto_plano
#Import das funções associadas ao manipulador usado
from pioneer_7dof import *

#criar um vetor coluna a partir de uma lista
def vetor(v):   
    return np.array([[v[0],v[1],v[2]]]).T

def MB_RRT(posicaod,q,erro_min,Kmax):

    n = getNumberJoints() #numero de juntas
    x  = vetor([1,0,0])

    destino = posicaod

    joint = getTypeJoints()

    #tamanho dos elos
    b = getLengthElos()

    K = Kmax #número máximo de iterações
    k = 0 #iteração inicial
    erromin = erro_min #erro minimo usado como um dos critérios de parada

    while(erro > erromin and k < K):