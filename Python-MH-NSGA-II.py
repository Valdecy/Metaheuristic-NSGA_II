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
import pandas as pd
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
    population = pd.DataFrame(np.zeros((population_size, len(min_values))))
    for i in range (0, len(list_of_functions)):
        name = str(i+1)
        name = "Fitness_" + name
        population[name] = 0.0
    for i in range(0, population_size):
        for j in range(0, len(min_values)):
             population.iloc[i,j] = random.uniform(min_values[j], max_values[j])      
        for k in range (1, len(list_of_functions) + 1):
            population.iloc[i,-k] = list_of_functions[-k](population.iloc[i,0:population.shape[1]-len(list_of_functions)])
    return population

# Function: Dominance
def dominance_function(solution_1, solution_2, number_of_functions = 2):
    count = 0
    dominance = True
    for k in range (1, number_of_functions + 1):
        if (solution_1.iloc[-k] <= solution_2.iloc[-k]):
            count = count + 1
    if (count == number_of_functions):
        dominance = True
    else:
        dominance = False       
    return dominance

# Function: Fast Non-Dominated Sorting
def fast_non_dominated_sorting(population, number_of_functions = 2):
    dominated_by = pd.DataFrame(np.zeros((population.shape[0], 1)), columns = ['Dominated_By'])
    dominated_by = dominated_by.applymap(str)    
    dominates_it = pd.DataFrame(np.zeros((population.shape[0], 1)), columns = ['Dominates_It'])
    dominates_it = dominates_it.applymap(str)       
    rank = pd.DataFrame(np.zeros((population.shape[0], 1)), columns = ['Dominance_N'])
    for i in range(0, population.shape[0]):
        for j in range(0, population.shape[0]):
            if(i != j):
                if dominance_function(solution_1 = population.iloc[i,:], solution_2 = population.iloc[j,:], number_of_functions = number_of_functions):
                    dominates_it.iloc[i,0] = str(dominates_it.iloc[i,0]) + ", " + str(j)
                    dominates_it.iloc[i,0] = dominates_it.iloc[i,0].replace("0.0, ", "")
                    dominates_it.iloc[i,0] = dominates_it.iloc[i,0].replace("None, ", "")
                    dominated_by.iloc[j,0] = str(dominated_by.iloc[j,0]) + ", " + str(i)
                    dominated_by.iloc[j,0] = dominated_by.iloc[j,0].replace("0.0, ", "")  
                    dominated_by.iloc[j,0] = dominated_by.iloc[j,0].replace("None, ", "")                    
                    if (rank.iloc[i,0] > 0):
                        rank.iloc[j,0] = rank.iloc[j,0] + 1
                    else:
                        rank.iloc[j,0] = 1
        dominated_by.iloc[i,0] = dominated_by.iloc[i,0].replace("0.0", "None")
        dominates_it.iloc[i,0] = dominates_it.iloc[i,0].replace("0.0", "None")        
    rank['Rank'] = rank['Dominance_N'].rank(method = 'dense') 
    return rank, dominated_by, dominates_it  

# Function: Sort Population by Rank
def sort_population_by_rank(population, rank):
    rank = rank.sort_values(by = 'Rank')
    rank_new = pd.DataFrame(np.zeros((population.shape[0], 2)), columns = ['Dominance_N', 'Rank'])
    population_new = population.copy(deep = True)  
    population_new = population_new.reset_index(drop=True)
    for i in range(0, population.shape[0]):
        idx = rank.index.values.astype(int)[i]
        rank_new.iloc[i,0] = rank.iloc[i,0] 
        rank_new.iloc[i,1] = rank.iloc[i,1]
        for k in range(0, population.shape[1]):
            population_new.iloc[i,k] = population.iloc[idx,k]
    return population_new, rank_new

# Function: Neighbour Sorting
def neighbour_sorting(population, rank, column = 0, index_value = 1, value = 0):
    sorted_population = population.loc[rank['Rank'] == index_value].copy(deep = True)
    sorted_population = sorted_population.sort_values(by = population.columns.values[column])
    sorted_population = sorted_population.reset_index(drop=True)
    value_lower = float("inf")
    value_upper = float("inf")
    for i in range(0, sorted_population.shape[0]):
        if (sorted_population.iloc[i, column] == value and sorted_population.shape[0] > 2):
            if (i == 0):
                value_lower = float("inf")
                value_upper = sorted_population.iloc[i+1, column] 
                break
            elif (i == sorted_population.shape[0] - 1):
                value_lower = sorted_population.iloc[i-1, column]
                value_upper = float("inf")
                break
            else:
                value_lower = sorted_population.iloc[i-1, column]
                value_upper = sorted_population.iloc[i+1, column]  
                break
    return value_lower, value_upper

# Function: Crowding Distance
def crowding_distance_function(population, rank, min_values = [-5,-5], max_values = [5,5], list_of_functions = [func_1, func_2]):
    crowding_distance = pd.DataFrame(np.zeros((population.shape[0], 1)), columns = ['Crowding_Distance'])    
    for i in range(0, population.shape[0]):
        for j in range(1, len(list_of_functions) + 1):
            f_minus_1, f_plus_1 = neighbour_sorting(population, rank, column = -j, index_value = rank.iloc[i, 1], value = population.iloc[i,-j]) 
            f_min_r, f_max_r = list_of_functions[-j](min_values), list_of_functions[-j](max_values)
            if (f_minus_1 == float("inf") or f_plus_1 == float("inf")):
                crowding_distance.iloc[i, 0] = 99999999999
            else:
                crowding_distance.iloc[i, 0] = crowding_distance.iloc[i, 0] + (f_plus_1 - f_minus_1)/(f_max_r - f_min_r + 1)
    return crowding_distance 

# Function:Crowded Comparison Operator
def crowded_comparison_operator(rank, crowding_distance, individual_1 = 0, individual_2 = 1):
    selection = False
    if (rank.iloc[individual_1,1] < rank.iloc[individual_2,1]) or ((rank.iloc[individual_1,1] == rank.iloc[individual_2,1]) and (crowding_distance.iloc[individual_1,0] > crowding_distance.iloc[individual_2,0])):
        selection = True      
    return selection

# Function: Offspring
def breeding(population, rank, crowding_distance, min_values = [-5,-5], max_values = [5,5], mu = 1, list_of_functions = [func_1, func_2]):
    offspring = population.copy(deep = True)
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
            parent_1 = i1         
        if (crowded_comparison_operator(rank, crowding_distance, individual_1 = i3, individual_2 = i4) == True):
            parent_2 = i3
        elif (crowded_comparison_operator(rank, crowding_distance, individual_1 = i4, individual_2 = i3) == True):
            parent_2 = i4
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
            offspring.iloc[i,j] = ((1 + b_offspring)*population.iloc[parent_1, j] + (1 - b_offspring)*population.iloc[parent_2, j])/2
            if (offspring.iloc[i,j] > max_values[j]):
                offspring.iloc[i,j] = max_values[j]
            elif (offspring.iloc[i,j] < min_values[j]):
                offspring.iloc[i,j] = min_values[j]            
            if(i < population.shape[0] - 1):   
                offspring.iloc[i + 1,j] = ((1 - b_offspring)*population.iloc[parent_1, j] + (1 + b_offspring)*population.iloc[parent_2, j])/2
                if (offspring.iloc[i + 1,j] > max_values[j]):
                    offspring.iloc[i + 1,j] = max_values[j]
                elif (offspring.iloc[i + 1,j] < min_values[j]):
                    offspring.iloc[i + 1,j] = min_values[j] 
        for k in range (1, len(list_of_functions) + 1):
            offspring.iloc[i,-k] = list_of_functions[-k](offspring.iloc[i,0:offspring.shape[1]-len(list_of_functions)])
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
                offspring.iloc[i,j] = offspring.iloc[i,j] + d_mutation
                if (offspring.iloc[i,j] > max_values[j]):
                    offspring.iloc[i,j] = max_values[j]
                elif (offspring.iloc[i,j] < min_values[j]):
                    offspring.iloc[i,j] = min_values[j]                     
        for k in range (1, len(list_of_functions) + 1):
            offspring.iloc[i,-k] = list_of_functions[-k](offspring.iloc[i,0:offspring.shape[1]-len(list_of_functions)])
    return offspring 

# NSGA II Function
def non_dominated_sorting_genetic_algorithm_II(population_size = 5, mutation_rate = 0.1, elite = 0, min_values = [-5,-5], max_values = [5,5], list_of_functions = [func_1, func_2], generations = 50, mu = 1, eta = 1):        
    count = 0
    number_of_functions = len(list_of_functions)    
    population = initial_population(population_size = population_size, min_values = min_values, max_values = max_values, list_of_functions = list_of_functions)  
    offspring = initial_population(population_size = population_size, min_values = min_values, max_values = max_values, list_of_functions = list_of_functions)  
    while (count <= generations):       
        print("Generation = ", count)
        population = pd.concat([population, offspring])
        rank, _ , _ = fast_non_dominated_sorting(population, number_of_functions = number_of_functions)
        population, rank = sort_population_by_rank(population, rank)
        population, rank = population.iloc[0:population_size,:], rank.iloc[0:population_size,:] 
        rank, _ , _ = fast_non_dominated_sorting(population, number_of_functions = number_of_functions)  
        population, rank = sort_population_by_rank(population, rank)   
        crowding_distance = crowding_distance_function(population, rank, min_values = min_values, max_values = max_values, list_of_functions = list_of_functions)
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

# Shaffer Pareto Front
schaffer = pd.DataFrame(np.arange(0.0, 2.0, 0.01))
schaffer['Function 1'] = 0.0
schaffer['Function 2'] = 0.0
for i in range (0, schaffer.shape[0]):
    schaffer.iloc[i,1] = schaffer_f1(variables_values = [schaffer.iloc[i,0]])
    schaffer.iloc[i,2] = schaffer_f2(variables_values = [schaffer.iloc[i,0]])

schaffer_1 = schaffer.iloc[:,1]
schaffer_2 = schaffer.iloc[:,2]

# Calling NSGA II Function
nsga_II_schaffer = non_dominated_sorting_genetic_algorithm_II(population_size = 40, mutation_rate = 0.1, min_values = [-5], max_values = [5], list_of_functions = [schaffer_f1, schaffer_f2], generations = 250, mu = 10, eta = 10)

# Graph Pareto Front Solutions
func_1_values = nsga_II_schaffer.iloc[:,-2]
func_2_values = nsga_II_schaffer.iloc[:,-1]
ax1 = plt.figure(figsize = (15,15)).add_subplot(111)
plt.xlabel('Function 1', fontsize = 12)
plt.ylabel('Function 2', fontsize = 12)
ax1.scatter(func_1_values, func_2_values, c = 'red',   s = 25, marker = 'o', label = 'NSGA-II')
ax1.scatter(schaffer_1,    schaffer_2,    c = 'black', s = 2,  marker = 's', label = 'Pareto Front')
plt.legend(loc = 'upper right');
plt.show()
