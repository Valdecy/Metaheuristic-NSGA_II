############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: NSGA-II

# Citation: 
# PEREIRA, V. (2018). Project: Metaheuristic-NSGA-II, File: Python-MH-NSGA-II.py, GitHub repository: <https://github.com/Valdecy/Metaheuristic-NSGA-II>

############################################################################

# Required Libraries
import numpy  as np
import math
import matplotlib.pyplot as plt
import random
import os

# Function 1
def func_1():
    return

# Function 2
def func_2():
    return

# Function: Initialize Variables
def initial_population(population_size = 5, min_values = [-5,-5], max_values = [5,5], list_of_functions = [func_1, func_2]):
    population = np.zeros((population_size, len(min_values) + len(list_of_functions)))
    for i in range(0, population_size):
        for j in range(0, len(min_values)):
             population[i,j] = random.uniform(min_values[j], max_values[j])      
        for k in range (1, len(list_of_functions) + 1):
            population[i,-k] = list_of_functions[-k](list(population[i,0:population.shape[1]-len(list_of_functions)]))
    return population

# Function: Dominance
def dominance_function(solution_1, solution_2, number_of_functions = 2):
    count = 0
    dominance = True
    for k in range (1, number_of_functions + 1):
        if (solution_1[-k] <= solution_2[-k]):
            count = count + 1
    if (count == number_of_functions):
        dominance = True
    else:
        dominance = False       
    return dominance

# Function: Fast Non-Dominated Sorting
def fast_non_dominated_sorting(population, number_of_functions = 2):
    S=[[] for i in range(0, population.shape[0])]
    front = [[]]
    n=[0 for i in range(0, population.shape[0])]
    rank = [0 for i in range(0, population.shape[0])]

    for p in range(0, population.shape[0]):
        S[p]=[]
        n[p]=0
        for q in range(0, population.shape[0]):
            if dominance_function(solution_1 = population[p,:], solution_2 = population[q,:], number_of_functions = number_of_functions):
                if q not in S[p]:
                    S[p].append(q)
            elif dominance_function(solution_1 = population[q,:], solution_2 = population[p,:], number_of_functions = number_of_functions):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)
    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)
    del front[len(front)-1]
    rank = np.zeros((population.shape[0], 1))
    for i in range(0, len(front)):
        for j in range(0, len(front[i])):
            rank[front[i][j], 0] = i + 1
    return rank

# Function: Sort Population by Rank
def sort_population_by_rank(population, rank):
    idx = np.argsort(rank, axis = 0)
    rank_new = np.zeros((population.shape[0], 1))
    population_new = np.zeros((population.shape[0], population.shape[1]))  
    for i in range(0, population.shape[0]):
        rank_new[i, 0] = rank[idx[i], 0] 
        for k in range(0, population.shape[1]):
            population_new[i,k] = population[idx[i],k]
    return population_new, rank_new

# Function: Neighbour Sorting
def neighbour_sorting(population, rank, column = 0, index_value = 1, value = 0):
    sorted_population = np.copy(population)
    for  i in range(rank.shape[0] -1, 0, -1):
        if (rank[i, 0] != index_value):
            sorted_population = np.delete(sorted_population, i, 0)
    sorted_population_ordered = (sorted_population[sorted_population[: ,column].argsort()])
    value_lower = float("inf")
    value_upper = float("inf")
    for i in range(0, sorted_population_ordered.shape[0]):
        if (sorted_population_ordered[i, column] == value and sorted_population_ordered.shape[0] > 2):
            if (i == 0):
                value_lower = float("inf")
                value_upper = sorted_population_ordered[i+1, column] 
                break
            elif (i == sorted_population_ordered.shape[0] - 1):
                value_lower = sorted_population_ordered[i-1, column]
                value_upper = float("inf")
                break
            else:
                value_lower = sorted_population_ordered[i-1, column]
                value_upper = sorted_population_ordered[i+1, column]  
                break
    return value_lower, value_upper

# Function: Crowding Distance
def crowding_distance_function(population, rank, number_of_functions = 2):
    crowding_distance = np.zeros((population.shape[0], 1))    
    for i in range(0, population.shape[0]):
        for j in range(1, number_of_functions + 1):
            f_minus_1, f_plus_1 = neighbour_sorting(population, rank, column = -j, index_value = rank[i, 0], value = population[i,-j])  
            if (f_minus_1 == float("inf") or f_plus_1 == float("inf")):
                crowding_distance[i, 0] = 99999999999
            else:
                crowding_distance[i, 0] = crowding_distance[i, 0] + (f_plus_1 - f_minus_1)
    return crowding_distance 

# Function:Crowded Comparison Operator
def crowded_comparison_operator(rank, crowding_distance, individual_1 = 0, individual_2 = 1):
    selection = False
    if (rank[individual_1,0] < rank[individual_2,0]) or ((rank[individual_1,0] == rank[individual_2,0]) and (crowding_distance[individual_1,0] > crowding_distance[individual_2,0])):
        selection = True      
    return selection

# Function: Offspring
def breeding(population, rank, crowding_distance, min_values = [-5,-5], max_values = [5,5], mu = 1, list_of_functions = [func_1, func_2]):
    offspring = np.copy(population)
    parent_1 = 0
    parent_2 = 1
    b_offspring = 0
    for i in range (0, offspring.shape[0]):
        i1, i2, i3, i4 = random.sample(range(0, len(population) - 1), 4)
        if (crowded_comparison_operator(rank, crowding_distance, individual_1 = i1, individual_2 = i2) == True):
            parent_1 = i1
        elif (crowded_comparison_operator(rank, crowding_distance, individual_1 = i2, individual_2 = i1) == True):
            parent_1 = i2
        else:
            rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            if (rand > 0.5):
                parent_1 = i1
            else:
                parent_1 = i2
        if (crowded_comparison_operator(rank, crowding_distance, individual_1 = i3, individual_2 = i4) == True):
            parent_2 = i3
        elif (crowded_comparison_operator(rank, crowding_distance, individual_1 = i4, individual_2 = i3) == True):
            parent_2 = i4
        else:
            rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            if (rand > 0.5):
                parent_2 = i3
            else:
                parent_2 = i4
        for j in range(0, offspring.shape[1] - len(list_of_functions)):
            rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            rand_b = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)                                
            if (rand <= 0.5):
                b_offspring = 2*(rand_b)
                b_offspring = b_offspring**(1/(mu + 1))
            elif (rand > 0.5):  
                b_offspring = 1/(2*(1 - rand_b))
                b_offspring = b_offspring**(1/(mu + 1))       
            offspring[i,j] = np.clip(((1 + b_offspring)*population[parent_1, j] + (1 - b_offspring)*population[parent_2, j])/2, min_values[j], max_values[j])           
            if(i < population.shape[0] - 1):   
                offspring[i+1,j] = np.clip(((1 - b_offspring)*population[parent_1, j] + (1 + b_offspring)*population[parent_2, j])/2, min_values[j], max_values[j]) 
        for k in range (1, len(list_of_functions) + 1):
            offspring[i,-k] = list_of_functions[-k](offspring[i,0:offspring.shape[1]-len(list_of_functions)])
    return offspring 

# Function: Mutation
def mutation(offspring, mutation_rate = 0.1, eta = 1, min_values = [-5,-5], max_values = [5,5], list_of_functions = [func_1, func_2]):
    d_mutation = 0            
    for i in range (0, offspring.shape[0]):
        for j in range(0, offspring.shape[1] - len(list_of_functions)):
            probability = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            if (probability < mutation_rate):
                rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
                rand_d = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)                                     
                if (rand <= 0.5):
                    d_mutation = 2*(rand_d)
                    d_mutation = d_mutation**(1/(eta + 1)) - 1
                elif (rand > 0.5):  
                    d_mutation = 2*(1 - rand_d)
                    d_mutation = 1 - d_mutation**(1/(eta + 1))                
                offspring[i,j] = np.clip((offspring[i,j] + d_mutation), min_values[j], max_values[j])                        
        for k in range (1, len(list_of_functions) + 1):
            offspring[i,-k] = list_of_functions[-k](offspring[i,0:offspring.shape[1]-len(list_of_functions)])
    return offspring 

# NSGA II Function
def non_dominated_sorting_genetic_algorithm_II(population_size = 5, mutation_rate = 0.1, min_values = [-5,-5], max_values = [5,5], list_of_functions = [func_1, func_2], generations = 50, mu = 1, eta = 1):        
    count = 0   
    population = initial_population(population_size = population_size, min_values = min_values, max_values = max_values, list_of_functions = list_of_functions)  
    offspring = initial_population(population_size = population_size, min_values = min_values, max_values = max_values, list_of_functions = list_of_functions)  
    while (count <= generations):       
        print("Generation = ", count)
        population = np.vstack([population, offspring])
        rank = fast_non_dominated_sorting(population, number_of_functions = len(list_of_functions))
        population, rank = sort_population_by_rank(population, rank)
        population, rank = population[0:population_size,:], rank[0:population_size,:] 
        rank = fast_non_dominated_sorting(population, number_of_functions = len(list_of_functions))
        population, rank = sort_population_by_rank(population, rank)
        crowding_distance = crowding_distance_function(population, rank,  number_of_functions = len(list_of_functions))
        offspring = breeding(population, rank, crowding_distance, mu = mu, min_values = min_values, max_values = max_values, list_of_functions = list_of_functions)
        offspring = mutation(offspring, mutation_rate = mutation_rate, eta = eta, min_values = min_values, max_values = max_values, list_of_functions = list_of_functions)             
        count = count + 1              
    return population

######################## Part 1 - Usage ####################################

# Schaffer Function 1
def schaffer_f1(variables_values = [0]):
    y = variables_values[0]**2
    return y

# Schaffer Function 2
def schaffer_f2(variables_values = [0]):
    y = (variables_values[0]-2)**2
    return y

# Calling NSGA II Function
nsga_II_schaffer = non_dominated_sorting_genetic_algorithm_II(population_size = 50, mutation_rate = 0.1, min_values = [-5], max_values = [5], list_of_functions = [schaffer_f1, schaffer_f2], generations = 100, mu = 10, eta = 10)

# Shaffer Pareto Front
schaffer = np.zeros((200, 3))
x = np.arange(0.0, 2.0, 0.01)
for i in range (0, schaffer.shape[0]):
    schaffer[i,0] = x[i]
    schaffer[i,1] = schaffer_f1(variables_values = [schaffer[i,0]])
    schaffer[i,2] = schaffer_f2(variables_values = [schaffer[i,0]])

schaffer_1 = schaffer[:,1]
schaffer_2 = schaffer[:,2]

# Graph Pareto Front Solutions
func_1_values = nsga_II_schaffer[:,-2]
func_2_values = nsga_II_schaffer[:,-1]
ax1 = plt.figure(figsize = (15,15)).add_subplot(111)
plt.xlabel('Function 1', fontsize = 12)
plt.ylabel('Function 2', fontsize = 12)
ax1.scatter(func_1_values, func_2_values, c = 'red',   s = 25, marker = 'o', label = 'NSGA-II')
ax1.scatter(schaffer_1,    schaffer_2,    c = 'black', s = 2,  marker = 's', label = 'Pareto Front')
plt.legend(loc = 'upper right')
plt.show()

######################## Part 2 - Usage ####################################

# Kursawe Function 1
def kursawe_f1(variables_values = [0, 0]):
    f1 = 0
    if (len(variables_values) == 1):
        f1 = f1 - 10 * math.exp(-0.2 * math.sqrt(variables_values[0]**2 + variables_values[0]**2))
    else:
        for i in range(0, len(variables_values)-1):
            f1 = f1 - 10 * math.exp(-0.2 * math.sqrt(variables_values[i]**2 + variables_values[i + 1]**2))
    return f1

# Kursawe Function 2
def kursawe_f2(variables_values = [0, 0]):
    f2 = 0
    for i in range(0, len(variables_values)):
        f2 = f2 + abs(variables_values[i])**0.8 + 5 * math.sin(variables_values[i]**3)
    return f2

# Calling NSGA II Function
nsga_II_kursawe = non_dominated_sorting_genetic_algorithm_II(population_size = 50, mutation_rate = 0.1, min_values = [-5,-5], max_values = [5,5], list_of_functions = [kursawe_f1, kursawe_f2], generations = 250, mu = 1, eta = 1)

# Kursawe Pareto Front
kursawe = np.zeros((10000, 4))
x = np.arange(-5, 5, 0.1)
count = 0
for j in range (0,100):
    for k in range (0, 100):
            kursawe[count,0] = x[j]
            kursawe[count,1] = x[k]
            count = count + 1
        
for i in range (0, kursawe.shape[0]):
    kursawe[i,2] = kursawe_f1(variables_values = [kursawe[i,0], kursawe[i,1]])
    kursawe[i,3] = kursawe_f2(variables_values = [kursawe[i,0], kursawe[i,1]])

kursawe_1 = kursawe[:,2]
kursawe_2 = kursawe[:,3]

# Graph Pareto Front Solutions
func_1_values = nsga_II_kursawe[:,-2]
func_2_values = nsga_II_kursawe[:,-1]
ax1 = plt.figure(figsize = (15,15)).add_subplot(111)
plt.xlabel('Function 1', fontsize = 12)
plt.ylabel('Function 2', fontsize = 12)
ax1.scatter(func_1_values, func_2_values, c = 'red',   s = 25, marker = 'o', label = 'NSGA-II')
ax1.scatter(kursawe_1,     kursawe_2,     c = 'black', s = 2,  marker = 's', label = 'Solutions')
plt.legend(loc = 'upper right')
plt.show()
