# -*- coding: utf-8 -*-
"""
Perceptron Simples
"""
import  numpy as np

#Define o npumero de épocas e o número de amostras (q)
numEpocas =5

q = 6

#Atributos
peso = np.array([113, 122, 107, 98, 115, 120])
pH = np.array([6.8, 4.7, 5.2, 3.6, 2.9, 4.2])

#Bias
bias = 1

#Entrada do perceptron
X = np.vstack((peso,pH))
Y = np.array([-1, 1, -1, -1, 1, 1]) 

#Taxa de aprendizado
eta = 0.1

#Defube i vetor de pesos
W = np.zeros([1,3])     #Duas entradas + o bias

#Array para armazenar os erros
e = np.zeros(6)

def funcaoAtivacao(valor):
    #A função de ativação a degrau bipolar
    if valor < 0.0:
        return (-1)
    else:
        return (1)

for i in range(numEpocas):
    for j in range (q):
        #Insere o bias no vetor de entrada
        Xb = np.hstack((bias, X[:,j]))
        
        #Calcula o campo induzido
        V = np.dot(W,Xb)
        
        #Calcular a saída do perceptron
        Yr = funcaoAtivacao(V)
        
        #Calcular o erro: e = (Y - Yr)
        e[j] = Y[j] - Yr
        
        #Treinamento do perceptron
        W = W + eta * e[j] * Xb
        
print("Vetor de erros (e) = " + str(e))        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        