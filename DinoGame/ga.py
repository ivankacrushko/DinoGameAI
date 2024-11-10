import numpy as np

def initialize_population(population_size, input_size, hidden_size, output_size):
    population = []
    for _ in range(population_size):
        # Tworzymy losowe wagi
        weights = {
            "W1": np.random.randn(hidden_size, input_size),
            "b1": np.random.randn(hidden_size, 1),
            "W2": np.random.randn(output_size, hidden_size),
            "b2": np.random.randn(output_size, 1)
        }
        population.append(weights)
    return population

def evaluate_fitness(agent, game):
    # Zak³adam, ¿e game.run(agent) zwraca wynik dla agenta w grze dino
    score = game.run(agent)
    return score

def select(population, fitness_scores, num_survivors):
    # Sortujemy populacjê wg wyników i wybieramy `num_survivors` najlepszych
    sorted_indices = np.argsort(fitness_scores)[::-1]
    survivors = [population[i] for i in sorted_indices[:num_survivors]]
    return survivors

def crossover(parent1, parent2):
    child = {}
    for key in parent1.keys():
        # Uœredniamy wagi obu rodziców
        child[key] = (parent1[key] + parent2[key]) / 2
    return child

def mutate(agent, mutation_rate=0.1):
    for key in agent.keys():
        # Losowa zmiana wartoœci wag z okreœlonym prawdopodobieñstwem
        mutation_mask = np.random.rand(*agent[key].shape) < mutation_rate
        agent[key] += mutation_mask * np.random.randn(*agent[key].shape)
    return agent

def create_new_generation(survivors, population_size):
    new_population = []
    num_survivors = len(survivors)
    while len(new_population) < population_size:
        parent1, parent2 = np.random.choice(survivors, 2)
        child = crossover(parent1, parent2)
        child = mutate(child)
        new_population.append(child)
    return new_population


def train_dino_agent(game, generations=100, population_size=50, num_survivors=10):
    # Rozmiary sieci
    input_size, hidden_size, output_size = 10, 20, 1  # Zmieñ na w³asne wartoœci
    # Inicjalizacja populacji
    population = initialize_population(population_size, input_size, hidden_size, output_size)
    
    for generation in range(generations):
        # Oceniamy agentów
        fitness_scores = [evaluate_fitness(agent, game) for agent in population]
        
        # Wybieramy najlepszych
        survivors = select(population, fitness_scores, num_survivors)
        
        # Tworzymy now¹ generacjê
        population = create_new_generation(survivors, population_size)
        
        # Wyœwietlamy wyniki
        best_score = max(fitness_scores)
        print(f"Pokolenie {generation + 1}, Najlepszy wynik: {best_score}")
