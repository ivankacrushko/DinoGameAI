
import pygame
import random
import numpy as np

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

import ann, game, gui, loading
from gui import Toolbar

pygame.init()

screenH = 600
screenW = 1100
screen = pygame.display.set_mode((screenW, screenH))

toolbar = Toolbar(screen, screenW)
toolbar.add_button("New")
toolbar.add_button("Open")
toolbar.add_button("Options")
toolbar.add_button("Reset gen")
toolbar.add_button("Quit")



highest_score = 0
death_count = 0

training_settings = loading.load_training_settings('neuron_training_config.txt')

def neuron_training(learning_rate, epochs, seed):
    file_path = 'train_data.txt'
    loaded_data_str = np.loadtxt(file_path, dtype=str)
    loaded_data = loaded_data_str.astype(int)
    train_X = loaded_data[:, :5]  
    train_Y = loaded_data[:, 5:]

    
    
    NN_ARCHITECTURE = loading.load_architecture_from_file('layers_config.txt')
    NN_ARCHITECTURE = loading.format_nn_architecture(NN_ARCHITECTURE)
    loading.load_game_settings("game_settings.txt")
    norm_train_X = []
    for row in train_X:
        X_norm = (row-row.mean())/row.std()   
        norm_train_X.append(X_norm)    
    norm_train_X = np.array(norm_train_X)
    
    gen = ann.Model()
    gen.train(np.transpose(train_X), np.transpose(train_Y), NN_ARCHITECTURE, epochs, learning_rate, seed)
    #(1000\0.0009)
    
    return gen

def filter_actions(model_output):
    if np.argmax(model_output) == 2:
        filtered_action = [False, False, True]
    if np.argmax(model_output) == 1:
        filtered_action = [False, True, False]
    if np.argmax(model_output) == 0:
        filtered_action = [True, False, False]
    
    return filtered_action

def detect_action(dino):
    actions = [dino.dino_run, dino.dino_duck, dino.dino_jump]
    if actions == [True, False, False]:
       return [1,0,0]
    elif actions == [False, True, False]:        
        return [0,1,0]
    elif actions == [False, False, True]:
        return [0,0,1]
    else:
        print("ERRROR") 


gen1 = neuron_training(training_settings['learning_rate'], training_settings['epochs'], 99)

def initialize_population(population_size, nn_architecture):
    population = []
    for _ in range(population_size):
        agent = ann.Model()  # Utwórz nowego agenta
        agent.init_layers(nn_architecture)  # Losowe wagi dla sieci
        population.append(agent)
    return population

def select(population, fitness_scores, num_survivors):
    sorted_indices = np.argsort(fitness_scores)[::-1]
    survivors = [population[i] for i in sorted_indices[:num_survivors]]
    return survivors

def crossover(parent1, parent2):
    child = ann.Model()
    child.params_values = {}
    for key in parent1.params_values.keys():
        child.params_values[key] = (parent1.params_values[key] + parent2.params_values[key]) / 2
    return child

def mutate(agent, mutation_rate=0.1):
    for key in agent.params_values.keys():
        mutation_mask = np.random.rand(*agent.params_values[key].shape) < mutation_rate
        agent.params_values[key] += mutation_mask * np.random.randn(*agent.params_values[key].shape)
    return agent

def create_new_generation(survivors, population_size, num_elites=1):
    new_population = []
    
    # Przeniesienie elity
    elites = survivors[:num_elites]
    new_population.extend(elites)
    
    # Tworzenie reszty populacji
    while len(new_population) < population_size:
        parent1, parent2 = np.random.choice(survivors, 2)
        child = crossover(parent1, parent2)
        child = mutate(child)
        new_population.append(child)
    
    return new_population

    
# NN_ARCHITECTURE = loading.load_architecture_from_file('layers_config.txt')
# NN_ARCHITECTURE = loading.format_nn_architecture(NN_ARCHITECTURE)

# # Inicjalizujemy populację oraz stan gry dla każdego agenta
# population = initialize_population(population_size, NN_ARCHITECTURE)
# players = [game.Dinosaur() for _ in range(population_size)]
# scores = [0] * population_size
# alive_agents = [True] * population_size  # Flaga, czy dany agent jest aktywny

def genetic_main():
    global game_speed, x_pos_bg, y_pos_bg,highest_score, points, death_count, num_survivors, generations, players, scores, population, population_size, alive_agents
    #was in global: obstacles, obstacle_distance, obstacle_width, obstacle_height,
    base_game_speed, jump_height, spawn_bats, spawn_high, spawn_short = loading.load_game_settings("game_settings.txt")
    run = True
    clock = pygame.time.Clock()  
    game_speed = base_game_speed
    
    population_size = 50
    num_survivors = 10
    num_elites = 1
    generations = 100

    # Inicjalizacja populacji oraz stanu gry
    NN_ARCHITECTURE = loading.load_architecture_from_file('layers_config.txt')
    NN_ARCHITECTURE = loading.format_nn_architecture(NN_ARCHITECTURE)
    population = initialize_population(population_size, NN_ARCHITECTURE)
    players = [game.Dinosaur() for _ in range(population_size)]
    scores = [0] * population_size
    alive_agents = [True] * population_size  # Flaga aktywności agenta
    generation = 1  # Licznik generacji
    
    
    for player in players:
        player.JUMP_VEL = jump_height

    cloud = game.Cloud(screenW)
    x_pos_bg = 0
    y_pos_bg = 380
    points = 0
    obstacles = []
    font = pygame.font.Font('freesansbold.ttf', 20)
    
    
    obstacle_distance = 0
    obstacle_width = 0
    obstacle_height = 0
    is_flying = -1
    
    def score(highest_score, generation, alive, points, game_speed):
        points += 1
        if points % 100 == 0:
            game_speed +=1            
        text = font.render("Score: " + str(points), True, (0,0,0))
        textRect = text.get_rect()
        textRect.center = (1000,40)
        screen.blit(text, textRect)

        text1 = font.render("Highest score: " + str(highest_score), True, (0,0,0))
        textRect = text1.get_rect()
        textRect.center = (1000,60)
        screen.blit(text1, textRect)
        
        text2 = font.render("Gen: " + str(generation), True, (0,0,0))
        textRect = text2.get_rect()
        textRect.center = (1000,80)
        screen.blit(text2, textRect)
        
        text3 = font.render("Alive: " + str(alive), True, (0,0,0))
        textRect = text2.get_rect()
        textRect.center = (1000, 100)
        screen.blit(text3, textRect)
        
        return game_speed

    def backgroundDraw():
        global x_pos_bg, y_pos_bg

        image_width = game.background.get_width()
        screen.blit(game.background, (x_pos_bg, y_pos_bg))
        screen.blit(game.background, (image_width + x_pos_bg, y_pos_bg))

        if x_pos_bg <= -image_width:
            screen.blit(game.background, (image_width + x_pos_bg, y_pos_bg))
            x_pos_bg = 0
        x_pos_bg -= game_speed    
    

    
    for generation in range(generations):
        players = [game.Dinosaur() for _ in range(population_size)]
        scores = [0] * population_size
        alive_agents = [True] * population_size
        points = 0
        obstacles = []
        game_speed = base_game_speed
        print(f"Pokolenie {generation + 1} rozpoczęte...")
        while any(alive_agents):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                toolbar.handle_event(event)

            screen.fill((255,255,255))
            userInput = pygame.key.get_pressed()
            toolbar.draw()

            if len(obstacles) == 0:
                if random.randint(0,2) == 0 and spawn_short == True:
                    obstacles.append(game.SmallCactus(game.smallCactus, screenW))
                    is_flying = -1
                elif random.randint(0,2) == 1 and spawn_high == True:
                    obstacles.append(game.LargeCactus(game.largeCactus, screenW))
                    is_flying = -1
                elif random.randint(0,2) == 2 and spawn_bats == True:
                    obstacles.append(game.Bird(game.bird, screenW))
                    is_flying = 1

            alives = 0
            for alive in alive_agents:
                if alive == True:
                    alives+=1
                
            for i, agent in enumerate(population):
                if alive_agents[i]:
                    inputs = np.array([[obstacle_distance, obstacle_height, obstacle_width, game_speed, is_flying]])
                
                    players[i].draw(screen)
                    model_output, _ = gen1.full_forward_propagation(np.reshape(inputs, (5,1)), agent.params_values, ann.NN_ARCHITECTURE)
                    players[i].update(userInput, filter_actions(model_output))
                
                    scores[i] += 1
            
                    for obstacle in obstacles:
                    
                        if players[i].dino_rect.colliderect(obstacle.rect):
                        
                            alive_agents[i] = False

            for obstacle in obstacles:
                obstacle.draw(screen)
                obstacle.update(game_speed, obstacles)
            try:
                obstacle_distance = obstacles[-1].rect.x
                obstacle_width = obstacles[-1].rect.width
                obstacle_height = obstacles[-1].rect.height
            except:
                pass

            if points > highest_score:
                highest_score = points    
        
            backgroundDraw()
            cloud.draw(screen)
            cloud.update(screenW, game_speed)
            game_speed = score(highest_score, generation, alives, points, game_speed)        
            clock.tick(30)
            pygame.display.update()
            
        fitness_scores = scores 
        survivors = select(population, fitness_scores, num_survivors)
        population = create_new_generation(survivors, population_size, num_elites)

        best_score = max(fitness_scores)
        print(f"Pokolenie {generation + 1} zakończone, Najlepszy wynik: {best_score}")

        generation += 1
    #menu(death_count, toolbar)

def backpropagation_main():
    global model, game_speed, x_pos_bg, y_pos_bg,highest_score, points, death_count
    #was in global: obstacles, obstacle_distance, obstacle_width, obstacle_height,
    game_speed, jump_height, spawn_bats, spawn_high, spawn_short = loading.load_game_settings("game_settings.txt")
    run = True
    clock = pygame.time.Clock()
    player = game.Dinosaur()   

    
    player.JUMP_VEL = jump_height

    cloud = game.Cloud(screenW)
    x_pos_bg = 0
    y_pos_bg = 380
    points = 0
    obstacles = []
    font = pygame.font.Font('freesansbold.ttf', 20)
    
    
    obstacle_distance = 0
    obstacle_width = 0
    obstacle_height = 0
    is_flying = -1
    
    def score(highest_score, death_count):
        global points, game_speed
        points += 1
        if points % 100 == 0:
            game_speed +=1            
        text = font.render("Score: " + str(points), True, (0,0,0))
        textRect = text.get_rect()
        textRect.center = (1000,40)
        screen.blit(text, textRect)

        text1 = font.render("Highest score: " + str(highest_score), True, (0,0,0))
        textRect = text1.get_rect()
        textRect.center = (1000,60)
        screen.blit(text1, textRect)
        
        text2 = font.render("Try: " + str(death_count+1), True, (0,0,0))
        textRect = text2.get_rect()
        textRect.center = (1000,80)
        screen.blit(text2, textRect)

    def backgroundDraw():
        global x_pos_bg, y_pos_bg

        image_width = game.background.get_width()
        screen.blit(game.background, (x_pos_bg, y_pos_bg))
        screen.blit(game.background, (image_width + x_pos_bg, y_pos_bg))

        if x_pos_bg <= -image_width:
            screen.blit(game.background, (image_width + x_pos_bg, y_pos_bg))
            x_pos_bg = 0
        x_pos_bg -= game_speed    
    

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            toolbar.handle_event(event)

        screen.fill((255,255,255))
        userInput = pygame.key.get_pressed()
        toolbar.draw()

        
        
        inputs = np.array([[obstacle_distance, obstacle_height, obstacle_width, game_speed, is_flying]])    

        player.draw(screen)
        gen1_forward, rr = gen1.full_forward_propagation(np.reshape(inputs, (5,1)), gen1.params_values, ann.NN_ARCHITECTURE)
        player.update(userInput, filter_actions(gen1_forward))



        if len(obstacles) == 0:
            if random.randint(0,2) == 0 and spawn_short == True:
                obstacles.append(game.SmallCactus(game.smallCactus, screenW))
                is_flying = -1
            elif random.randint(0,2) == 1 and spawn_high == True:
                obstacles.append(game.LargeCactus(game.largeCactus, screenW))
                is_flying = -1
            elif random.randint(0,2) == 2 and spawn_bats == True:
                obstacles.append(game.Bird(game.bird, screenW))
                is_flying = 1

        for obstacle in obstacles:
            obstacle.draw(screen)
            obstacle.update(game_speed, obstacles)
            if player.dino_rect.colliderect(obstacle.rect):
                death_count += 1
                menu(death_count, toolbar)
        try:
            obstacle_distance = obstacles[-1].rect.x
            obstacle_width = obstacles[-1].rect.width
            obstacle_height = obstacles[-1].rect.height
        except:
            pass

        if points > highest_score:
            highest_score = points    
        
        backgroundDraw()
        cloud.draw(screen)
        cloud.update(screenW, game_speed)
        score(highest_score, death_count)        
        clock.tick(30)
        pygame.display.update()

def menu(death_count, toolbar):
    global points, num_survivors, population_size, generation, scores, population
    run = True
            
    while run:
        screen.fill((255,255,255))
        font = pygame.font.Font('freesansbold.ttf', 20)
        if death_count == 0:
            text = font.render("Press any key to start", True, (0,0,0))
        elif death_count > 0:
            text = font.render("Press any key to restart", True, (0,0,0))
            score = font.render("Score: " +str(points), True, (0,0,0))
            deaths = font.render("Try: " +str(death_count), True, (0,0,0))
           #  fitness_scores = scores  # Wyniki gry traktujemy jako fitness
           #  survivors = select(population, fitness_scores, num_survivors)
           #  population = create_new_generation(survivors, population_size)

           #  best_score = max(fitness_scores)
           # # print(f"Pokolenie {1}, Najlepszy wynik: {best_score}")

            scoreRect = score.get_rect()
            scoreRect.center = (screenW //2, screenH // 2 + 50)
            screen.blit(score, scoreRect)
            deathRect = deaths.get_rect()
            deathRect.center = (screenW //2, screenH // 2 + 70)
            screen.blit(deaths, deathRect)
        textRect = text.get_rect()
        textRect.center = (screenW // 2, screenH // 2)
        screen.blit(text, textRect)
        toolbar.draw()
        pygame.display.update()
        
        for event in pygame.event.get():
            if event.type ==pygame.QUIT:
                run = False
                pygame.quit()
            if toolbar.handle_event(event) == True:
                gen1 = neuron_training()
            if event.type == pygame.KEYDOWN:
                if training_settings['method'] == 'Backpropagation':
                    backpropagation_main()
                else:
                    genetic_main()
                    


menu(death_count, toolbar)