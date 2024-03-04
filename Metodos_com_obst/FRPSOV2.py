#Autor David Oliveira.
#Mestrando em Engenharia Elétrica da Universidade Federal de Campina Grande-UFCG.
#Membro do Grupo de Pesquisa em Robotica da UFS-GPRUFS.
#Implementação do Fully Resampled Particle Swarm Optimizarion

#Import das bibliotecas python
from random import uniform
import numpy as np

#Import das minhas funções
from funcoes import deteccao_de_colisao, distancia, orientacao
#Import das funções associadas ao manipulador usado
from pioneer_7dof import *

class particle:
    def __init__(self,position,dimension):
        self.p = position #posição atual da particula/configuração do robô
        self.n = dimension #dimensão da particula
        self.d = 0 #Diferença em módulo da distância atual para a desejada
        self.o = np.array([0,0,0]) #Diferença em módulo da orientacao atual para a desejada
        self.f = np.Inf #Função de custo/fitnees atual da particula
                   
    def update_fuction(self,position,orientation,esferas): #Calcula a função de custo/fitness da particula
        #(posição,orientacao) da pose desejada
        limits = getLimits()
        for (qi,li) in zip(self.p, limits):
            if(np.abs(qi) > np.abs(li)):
                self.f = np.Inf
                return 

        #p,orient = Cinematica_Direta3(self.p)
        pontos, orient = Cinematica_Direta(self.p,True)
        end = np.shape(pontos)[1] -1

        for esfera in esferas:
            for i in range(end):
                r = esfera.get_raio() + getRaio()
                if(deteccao_de_colisao(pontos[:,i],pontos[:,i+1],esfera.get_centro(),r)):
                    self.f = np.Inf 
                    return

        
        p = pontos[:,end]
        #calculo do erro em módulo da orientacao desejada e da particula
        self.o = distancia(orientacao(orient),orientation,3)

        #calculo da distancia euclidiana da posição do efetuador em relação ao objetivo
        self.d = distancia(p,position,3)

        #Calculo da função de custo       
        k1 = 0.1 #orientacao
        k2 = 1 #posição

        self.f = (k1*self.o) + (k2*self.d)

def _FRPSO(position,orientation,number,n,L,erro_min,Kmax,esferas):
    #numero limite de interações
    k = Kmax     
    q = []
    Nbests = 1
    tau = 0.5#0.5

    b = 0.99
    a = (1 - 0.99)/k

    evolucao_qbets = []

    #criando as particulas de dimensão n e calculando o valor de sua função de custo
    for i in range(number):
        p = []
        for i2 in range(n):
            p.append(uniform(-L[i2],L[i2]))

        q.append(particle(p,n))
        q[i].update_fuction(position,orientation,esferas)

    #Criando as configurações qbests e sua funções de custos
    qbests = []
    qvalues = []
    f = np.inf
    for i in range(Nbests):      
        qbests.append(q[i].p.copy())
        qvalues.append(q[i].f)
        if(f <  q[i].f):
            f = q[i].f

    for i in range(number):
        if(max(qvalues) > q[i].f):
            for j in range(Nbests):
                if(qvalues[j] == max(qvalues)):
                    qvalues[j] = q[i].f
                    qbests[j] = q[i].p.copy()
                    break
            f = min(qvalues)

    qbests = np.array(qbests)
    qvalues = np.array(qvalues)
    idcs = np.argsort(qvalues)
    qvalues = qvalues[idcs]
    qbests = qbests[idcs]
    qBest = qbests[0]
    f = qvalues[0]

    #Executando FRPSO
    for j in range(k):
        q = []

        sig = f/tau
        for N in range(Nbests):
            for i in range(int(0.835*number)):
            #for i in range(int(number/(Nbests +1))):
                p = sig*np.random.randn(n)
            
                for i2 in range(n):
                    p[i2] = qbests[N][i2] + p[i2]
                q.append(particle(p,n))

        for i in range(number - len(q)):
            p = []
            for i2 in range(n):
                p.append(uniform(-L[i2],L[i2]))
            q.append(particle(p,n))

        c = a*j + b

        for i in range(number):          
            q[i].update_fuction(position,orientation,esferas)
            if(q[i].f < c*f):
                idc = np.argmax(qvalues)
                qbests[idc] = q[i].p.copy()
                qvalues[idc] = q[i].f
                qBest = q[i].p.copy()
                f = q[i].f
                break

        idcs = np.argsort(qvalues)
        qvalues = qvalues[idcs]
        qbests = qbests[idcs]
        
        #Critério de parada
        evolucao_qbets.append(f)
        if(f <= erro_min):
            break;   

    return [f,j+1,qBest,evolucao_qbets]

def FRPSOV2(posicaod,orientacaod,erro_min,Kmax,esferas):

    orientacaod = orientacao(orientacaod)
    numero_particulas = 1000
    dimensao = getNumberJoints() #dimensão do robô
    #restrições de cada ângulo
    L = getLimits()

    f,k,qBest,evolucao_qbets = _FRPSO(posicaod,orientacaod,numero_particulas,dimensao,L,erro_min,Kmax,esferas)
    #print(f)
    #plot(qBest,posicaod,esferas)

    return [f,k,evolucao_qbets]