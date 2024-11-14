
import pygame
import random
import numpy as np

import ann, game, loading

def filter_actions(model_output):
    if np.argmax(model_output) == 2:
        filtered_action = [False, False, True]
    if np.argmax(model_output) == 1:
        filtered_action = [False, True, False]
    if np.argmax(model_output) == 0:
        filtered_action = [True, False, False]
        
    return filtered_action

def neuron_training():
    training_settings = loading.load_training_settings('neuron_training_config.txt')
    learning_rate = training_settings['learning_rate']
    epochs = training_settings['epochs']
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
    gen.train(np.transpose(train_X), np.transpose(train_Y), NN_ARCHITECTURE, epochs, learning_rate, 99)
    #(1000\0.0009)
    
    return gen

highest_score = 0
death_count = 0


    

def backpropagation_main(screen, screenW, toolbar, gen1):
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
            if toolbar.handle_event(event) == "Quit":
                    return

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
                return
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