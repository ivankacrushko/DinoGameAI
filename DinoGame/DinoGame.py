
import pygame
import random
import numpy as np

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

import ann, game
from gui import Toolbar





pygame.init()

screenH = 600
screenW = 1100
screen = pygame.display.set_mode((screenW, screenH))

toolbar = Toolbar(screen, screenW)
toolbar.add_button("New")
toolbar.add_button("Open")
toolbar.add_button("Options")
toolbar.add_button("Quit")


highest_score = 0
death_count = 0

file_path = 'train_data.txt'
loaded_data_str = np.loadtxt(file_path, dtype=str)
loaded_data = loaded_data_str.astype(int)
train_X = loaded_data[:, :5]  
train_Y = loaded_data[:, 5:]  

norm_train_X = []
for row in train_X:
    X_norm = (row-row.mean())/row.std()   
    norm_train_X.append(X_norm)    
norm_train_X = np.array(norm_train_X)

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

gen1 = ann.Model()
gen1.train(np.transpose(train_X), np.transpose(train_Y), ann.NN_ARCHITECTURE, 1000, 0.0009)



def main():
    global model, game_speed, x_pos_bg, y_pos_bg,highest_score, points,obstacles, obstacle_distance, obstacle_width, obstacle_height, death_count
    run = True
    clock = pygame.time.Clock()
    player = game.Dinosaur()   

    

    cloud = game.Cloud(screenW)
    game_speed = 20
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
        text = font.render("Punkty: " + str(points), True, (0,0,0))
        textRect = text.get_rect()
        textRect.center = (1000,40)
        screen.blit(text, textRect)

        text1 = font.render("Rekord: " + str(highest_score), True, (0,0,0))
        textRect = text1.get_rect()
        textRect.center = (1000,60)
        screen.blit(text1, textRect)
        
        text2 = font.render("Podejscie: " + str(death_count+1), True, (0,0,0))
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

        
        ####
        inputs = np.array([[obstacle_distance, obstacle_height, obstacle_width, game_speed, is_flying]])    

        player.draw(screen)
        gen1_forward, rr = gen1.full_forward_propagation(np.reshape(inputs, (5,1)), gen1.params_values, ann.NN_ARCHITECTURE)
        player.update(userInput, filter_actions(gen1_forward))

        ####

        if len(obstacles) == 0:
            if random.randint(0,2) == 0:
                obstacles.append(game.SmallCactus(game.smallCactus, screenW))
                is_flying = -1
            elif random.randint(0,2) == 1:
                obstacles.append(game.LargeCactus(game.largeCactus, screenW))
                is_flying = -1
            elif random.randint(0,2) == 2:
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
    global points
    run = True
            
    while run:
        screen.fill((255,255,255))
        font = pygame.font.Font('freesansbold.ttf', 20)
        if death_count == 0:
            text = font.render("Nacisnij dowolny klawisz aby zaczac", True, (0,0,0))
        elif death_count > 0:
            text = font.render("Nacisnij dowolny klawisz aby ponowic", True, (0,0,0))
            score = font.render("Zdobyto punktow: " +str(points), True, (0,0,0))
            deaths = font.render("Podejscie: " +str(death_count), True, (0,0,0))
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
            toolbar.handle_event(event)
            if event.type == pygame.KEYDOWN:
                main()
                    

    
menu(death_count, toolbar)