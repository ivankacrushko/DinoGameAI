
import pygame

import loading
import genetic_algorithm as ga
import backpropagation as bp
from gui import Toolbar

pygame.init()

screenH = 600
screenW = 1100
screen = pygame.display.set_mode((screenW, screenH))
pygame.display.set_caption('Dino Game')

toolbar = Toolbar(screen, screenW)
toolbar.add_button("Options")
toolbar.add_button("Reset gen")
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
                pygame.quit()
                run = False
            if toolbar.handle_event(event) == "Reset":
                gen1 = bp.neuron_training()
            if toolbar.handle_event(event) == "Quit":
                pygame.quit()
                run = False
            if event.type == pygame.KEYDOWN:
                if training_settings['method'] == 'Backpropagation':
                    
                    gen1 = bp.neuron_training()
                    bp.backpropagation_main(screen, screenW, toolbar, gen1)
                else:
                    ga.genetic_main(screen, screenW, toolbar)
                    


menu(death_count, toolbar)