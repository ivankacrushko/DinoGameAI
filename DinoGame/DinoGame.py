
import pygame
import random
import numpy as np

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

import ann, game, gui, loading, ga, bp
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
toolbar.add_button("Menu")
toolbar.add_button("Quit")



highest_score = 0
death_count = 0

training_settings = loading.load_training_settings('neuron_training_config.txt')


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
        training_settings = loading.load_training_settings('neuron_training_config.txt')
        for event in pygame.event.get():
            if event.type ==pygame.QUIT:
                run = False
                pygame.quit()
            if toolbar.handle_event(event) == True:
                gen1 = bp.neuron_training(99)
            if event.type == pygame.KEYDOWN:
                if training_settings['method'] == 'Backpropagation':
                    gen1 = bp.neuron_training(99)
                    bp.backpropagation_main(screen, screenW, toolbar, gen1)
                else:
                    ga.genetic_main(screen, screenW, toolbar)
                    


menu(death_count, toolbar)