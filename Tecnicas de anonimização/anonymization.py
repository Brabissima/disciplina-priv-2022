import pandas as pd
import numpy as np
import random
import json
from collections import defaultdict
from sklearn import datasets
from multiprocessing.dummy import Pool
from datetime import datetime
from scipy.stats import laplace
import warnings

warnings.filterwarnings("ignore")

def suppression(features):
    for feature in features:
        dataset[feature].replace(dataset[feature].unique(), '*', inplace=True)
    return dataset


def generalization(level_local=0, level_bday=0):
    dataset = generalizing_local(level_local)
    dataset = generalizing_bday(dataset, level_bday)
    
    return dataset

def generalizing_local(level=0):
    if(level == 1):
        for i in range(len(dataset)):
            neigh, county, m_region, state = dataset['localCaso'].iloc[i].split('; ')
            dataset['localCaso'].iloc[i] = '*; {}; {}; {}'.format(county, m_region, state)
    elif(level == 2):
        for i in range(len(dataset)):
            neigh, county, m_region, state = dataset['localCaso'].iloc[i].split('; ')
            dataset['localCaso'].iloc[i] = '*; *; {}; {}'.format(m_region, state)
    elif(level == 3):
        for i in range(len(dataset)):
            neigh, county, m_region, state = dataset['localCaso'].iloc[i].split('; ')
            dataset['localCaso'].iloc[i] = '*; *; *; {}'.format(state)
    
    return dataset

def generalizing_bday(dataset, level=0):
    dataset = dataset.copy()
    dataset = dataset
    if(level == 1):
        for i in range(len(dataset)):
            day, month, year = dataset['dataNascimento'].iloc[i].split('/')
            dataset['dataNascimento'].iloc[i] = '*/{}/{}'.format(month, year)
    elif(level == 2):
        for i in range(len(dataset)):
            day, month, year = dataset['dataNascimento'].iloc[i].split('/')
            dataset['dataNascimento'].iloc[i] = '*/*/{}'.format(year)
    
    return dataset

def adding_laplace_noise(index, value):
    eps_n = eps/len(dataset)
    noise = laplace.rvs(loc=0, scale = sensitivity/eps_n)
    value = np.ceil(round(value + noise, 1)) # remove ceil(*) for float values\n",
   
    return index, value

def perturbation(feature):
    global sensitivity
    global eps
    
    if(isinstance(dataset[feature].iloc[0], str)):
        values, counts = dataset[feature].value_counts().index, dataset[feature].value_counts().values
        dataset[feature] = random.choices(values, weights=counts, k=len(dataset[feature])) # following the real frequencies distribution
    else:
        index_feature_2D = np.c_[np.array(dataset.index), np.array(dataset[feature])]
        eps = 75*len(dataset)
        sensitivity = max(dataset[feature]) - min(dataset[feature])
        with Pool(15) as pool:
            threads = pool.starmap(adding_laplace_noise, index_feature_2D)
            pool.terminate()
            pool.close()
        pool.join()
        
        dataset[feature] = [value for value in list(dict(threads).values())]
    
    return dataset

df = pd.read_csv('covid_com_meso.csv')
print("Tamanho do dataset: ",len(df))
n = int(input('Para quantas amostras deseja executar as operações?\n'))
global dataset
global pre_df
dataset = df[:n]
dataset.drop(columns='Unnamed: 0', axis=1, inplace=True)
pre_df = dataset.copy()


def menu():
    u = '#'*20
    print('\n\n{} Menu {}\n\n1 - Carregar dados pré-processados\n2 - Supressão\n3 - Generalização\n4 - Perturbação\n5 - Visualizar dados anonimizados\n6 - Calcular estatísticas pós-perturbação\n7 - Sair\n\n'.format(u, u))
    op = input('\nDigite a operação que deseja realizar:\n')
    return op

def op_1():
    print(pre_df)

def op_2():
    def menu_features():
        print('Selecione os atributos que deseja suprimir (separe os atributos com vírgula+espaço: ", "):\
               \n1 - Local\n2 - Gênero\n3 - Data do nascimento\n4 - Idade\n5 - Raça/Cor\n6 - Resultado do exame\n\n')
        features = input()
        return features
    
    global dataset
    features = menu_features()
    features = features.split(', ')
    attr = {'1': 'localCaso',
            '2': 'sexoCaso',
            '3': 'dataNascimento',
            '4': 'idadeCaso',
            '5': 'racaCor',
            '6': 'resultadoFinalExame'
           }
    
    flag = True
    for feat in features:
        if(feat not in attr.keys()):
            flag = False
    if(flag == True):
        dataset = suppression([attr[i] for i in features])
        print(dataset)
    else:
        print("\nValores inválidos.\n")
        op_2()
        
def op_3():
    global dataset
    print('\nSelecione o nível de generalização desejado para local, com 0 sendo o menor nivel e 3 o maior:\n')
    level_local = int(input())
    print('\nAgora selecione o nível de generalização desejado para nascimento, com 0 sendo o menor nível e 2 o maior:\n')
    level_bday = int(input())
    dataset = generalization(level_local, level_bday)
    print(dataset)
    
def op_4():
    def menu_feature():
        print('Selecione o atributo que deseja aplicar a perturbação:\
               \n1 - Local\n2 - Gênero\n3 - Data do nascimento\n4 - Idade\n5 - Raça/Cor\n6 - Resultado do exame\n\n')
        feature = input()
        return feature
    
    global dataset
    feature = menu_feature()
    attr = {'1': 'localCaso',
            '2': 'sexoCaso',
            '3': 'dataNascimento',
            '4': 'idadeCaso',
            '5': 'racaCor',
            '6': 'resultadoFinalExame'
           }
    
    dataset = perturbation(attr[feature])
    print(dataset)
    

def options():
    op = menu()

    if(op == '1'):
        op_1()
        options()

    elif(op == '2'):
        op_2()
        options()

    elif(op == '3'):
        op_3()
        options()
        
    elif(op == '4'):
        op_4()
        options()
    
    elif(op == '5'):
        print(dataset)
        options()
        
    elif(op == '6'):
        print("Estatísticas idade perturbada e não perturbada:\n\nMédia: {}, {}\nMediana: {}, {}\nDesvio Padrão: {}, {}".format(np.mean(dataset['idadeCaso']), np.mean(pre_df['idadeCaso']),
                                                             np.median(dataset['idadeCaso']), np.median(pre_df['idadeCaso']),
                                                             np.std(dataset['idadeCaso']), np.std(pre_df['idadeCaso'])))
    else:
        return
    
options()
dataset.to_csv('covid_publico.csv')