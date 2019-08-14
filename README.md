# Metaheuristic-NSGA_II
NSGA II (Non-Dominated Sorting Genetic Algorithm II) Function to Minimize Multiple Objectives with Continuous Variables. Real Values Encoded. The function returns: 1) An array containing the used value(s) for each function and the output for each function f(x). For example, if the functions f(x1, x2) and g(x1, x2) are used in this same order, then the array would be [x1, x2, f(x1, x2), g(x1, x2)]. 

* population_size = The population size. The Default Value is 5.

* mutation_rate = Chance to occur a mutation operation. The Default Value is 0.1

* eta = Value of the mutation operator. The Default Value is 1.

* min_values = The minimum value that the variable(s) from a list can have. The default value is -5.

* max_values = The maximum value that the variable(s) from a list can have. The default value is  5.

* generations = The total number of iterations. The Default Value is 50.

* list_of_functions = A list of functions. The default value is two fucntions [func_1, func_2].

* mu = Value of the breed operator. The Default Value is 1.

Kursawe Function Example:
<p align="center"> 
<img src="https://github.com/Valdecy/Metaheuristic-NSGA_II/blob/master/Python-MH-NSGA-II.gif">
</p
