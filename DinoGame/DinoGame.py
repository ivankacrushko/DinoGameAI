
from email.mime import image
from turtle import Screen, screensize
import pygame
import os
import random
import numpy as np
###
from keyboard import press, release
import pyautogui
import time


pygame.init()

screenH = 600
screenW = 1100
screen = pygame.display.set_mode((screenW, screenH))
epoka = 0

death_count = 0
total_reward = 0
reward = 0



running = [pygame.image.load(os.path.join("Assets/Dino", "DinoRun1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoRun2.png"))]
jumping = pygame.image.load(os.path.join("Assets/Dino", "DinoJump.png"))
ducking = [pygame.image.load(os.path.join("Assets/Dino", "DinoDuck1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoDuck2.png"))]
smallCactus = [pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus1.png")),
               pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus2.png")),
               pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus3.png"))]
largeCactus = [pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus1.png")),
               pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus2.png")),
               pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus3.png"))]
bird = [pygame.image.load(os.path.join("Assets/Bird", "Bird1.png")),
        pygame.image.load(os.path.join("Assets/Bird", "Bird2.png"))]
cloud = pygame.image.load(os.path.join("Assets/Other", "Cloud.png"))
background = pygame.image.load(os.path.join("Assets/Other", "Track.png"))

class Dinosaur:
    xPos = 80
    yPos = 310
    yPosDuck = 340
    jumpVelocity = 8.5

    def __init__(self):
        self.duck_img = ducking
        self.run_img = running
        self.jump_img = jumping

        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False

        self.step_index = 0
        self.jump_v = self.jumpVelocity
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.xPos
        self.dino_rect.y = self.yPos

    def update(self, userInput):
        if self.dino_duck:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()

        if self.step_index >= 10:
            self.step_index = 0

        if userInput[pygame.K_UP] and not self.dino_jump:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
        elif userInput[pygame.K_DOWN] and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
        elif not (self.dino_jump or userInput[pygame.K_DOWN]):
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False

    def duck(self):
        self.image = self.duck_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.xPos
        self.dino_rect.y = self.yPosDuck
        self.step_index += 1

    def run(self):
        self.image = self.run_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.xPos
        self.dino_rect.y = self.yPos
        self.step_index += 1

    def jump(self):
        self.image = self.jump_img
        if self.dino_jump:
            self.dino_rect.y -= self.jump_v * 4
            self.jump_v -= 0.8
        if self.jump_v < - self.jumpVelocity:
            self.dino_jump = False
            self.jump_v = self.jumpVelocity

    def draw(self, screen):
        screen.blit(self.image, (self.dino_rect.x, self.dino_rect.y))

class Cloud:
    def __init__(self):
        self.x = screenW + random.randint(800, 1000)
        self.y = random.randint(50,100)
        self.image = cloud
        self.width = self.image.get_width()

    def update(self):
         self.x -= game_speed
         if self.x < -self.width:
             self.x = screenW + random.randint(2500, 3000)
             self.y = random.randint(50,100)

    def draw(self, screen):
        screen.blit(self.image, (self.x, self.y))

class Obstacle:
    def __init__(self, image, type):
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()
        self.rect.x = screenW

    def update(self):
        self.rect.x -= game_speed
        
        if self.rect.x < -self.rect.width:
            obstacles.pop()


    def draw(self, screen):
        screen.blit(self.image[self.type], self.rect)

class SmallCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0,2)
        super().__init__(image,self.type)
        self.rect.y = 325

class LargeCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0,2)
        super().__init__(image,self.type)
        self.rect.y = 300

class Bird(Obstacle):
    def __init__(self, image):
        self.type = 0
        super().__init__(image,self.type)
        self.rect.y = 250
        self.index = 0

    def draw(self, screen):
        if self.index >= 9:
            self.index = 0
        screen.blit(self.image[self.index//5], self.rect)
        self.index += 1

class Model:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicjalizacja wag i obci¹¿eñ dla warstwy ukrytej
        self.weights_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

    def forward(self, inputs):
        # Propagacja wprzód z funkcj¹ aktywacji ReLU
        self.hidden_layer_output = relu(np.dot(inputs, self.weights_hidden) + self.bias_hidden)

        # Propagacja wprzód z funkcj¹ aktywacji softmax
        self.output = softmax(np.dot(self.hidden_layer_output, self.weights_output) + self.bias_output)

    def calculate_loss(self, predicted, target):
        epsilon = 1e-15
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        loss = -np.sum(target * np.log(predicted)) / len(target)
        return loss

    def backward(self, inputs, target):
        output_error = self.output - target

        weights_output_gradient = np.dot(self.hidden_layer_output.T, output_error)
        bias_output_gradient = np.sum(output_error, axis=0, keepdims=True)

        hidden_error = np.dot(output_error, self.weights_output.T)
        hidden_error[self.hidden_layer_output <= 0] = 0

        weights_hidden_gradient = np.dot(inputs.T, hidden_error)
        bias_hidden_gradient = np.sum(hidden_error, axis=0, keepdims=True)

        return weights_hidden_gradient, bias_hidden_gradient, weights_output_gradient, bias_output_gradient

    def update_weights(self, learning_rate, weights_hidden_gradient, bias_hidden_gradient, weights_output_gradient, bias_output_gradient):
        self.weights_hidden -= learning_rate * weights_hidden_gradient
        self.bias_hidden -= learning_rate * bias_hidden_gradient

        self.weights_output -= learning_rate * weights_output_gradient
        self.bias_output -= learning_rate * bias_output_gradient

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    probabilities = 1 / (1 + np.exp(-x))
    #print(probabilities)
    return probabilities

def reward_system(points, death_count):
    # Przyk³adowa logika funkcji nagrody
    
    reward = (points/10) - death_count/3
    #print(reward)
    return reward
    
def standardize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
    return standardized_data

def main2(model):
    global total_reward, reward, death_count
    #inputs
    learning_rate = 0.1

    normalized_inputs = np.array([0,0,0,0])
    target_options = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1],[1,0,0]])
   
    
    print(normalized_inputs)

    for i in range(len(normalized_inputs)):
            inputs = normalized_inputs[i]
            target = target_options[i]


            model.forward(inputs)           
            reward = reward_system(points, death_count)
            loss = model.calculate_loss(model.output, target) - reward
            
            gradients = model.backward(inputs, target)

            model.update_weights(learning_rate, *gradients)

            total_reward = total_reward + reward

    #print(model.weights_output)
    #print(gradients)
    print("nagroda: "+str(reward))
    print("cala nagroda: "+str(total_reward))
    print("smierci: "+str(death_count))
    #print(action)

def main():
    global game_speed, x_pos_bg, y_pos_bg, points,death_count, obstacles, obstacle_distance, obstacle_width, obstacle_height

    run = True
    clock = pygame.time.Clock()
    player = Dinosaur()
    cloud = Cloud()
    game_speed = 14
    x_pos_bg = 0
    y_pos_bg = 380
    points = 0
    obstacles = []
    
    obstacle_distance = 0
    obstacle_width = 0
    obstacle_height = 0

    font = pygame.font.Font('freesansbold.ttf', 20)

    model = Model(4,4,3)
    main2(model)

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        screen.fill((255,255,255))
        userInput = pygame.key.get_pressed()
        
        player.draw(screen)
        player.update(userInput)
        
        ####
        inputs = np.array([[obstacle_distance, obstacle_height, obstacle_width, game_speed]])
        
        
        normalized_inputs = np.array([(obstacle_distance - 300)/(150-300),
                         (obstacle_height - 65)/(97-65),
                         (obstacle_width-40)/(105-40),
                         (game_speed - 14)/(100-14)])

        target_options = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1],[1,0,0]])

        model.forward(inputs)
        action = np.argmax(model.output)
        
        
         

        
        #print(action)
        #print(np.argmax(model.output))

        if action == 0:
            player.dino_duck = False
            player.dino_run = True
            player.dino_jump = False
            player.run
            
        elif action == 1:
                player.dino_duck = True
                player.dino_run = False
                player.dino_jump = False
                player.duck
        elif action == 2:
                player.dino_duck = False
                player.dino_run = False
                player.dino_jump = True
                player.jump

        ####

        if len(obstacles) == 0:
            if random.randint(0,2) == 0:
                obstacles.append(SmallCactus(smallCactus))
            elif random.randint(0,2) == 1:
                obstacles.append(LargeCactus(largeCactus))
            elif random.randint(0,2) == 2:
                obstacles.append(Bird(bird))
        

        for obstacle in obstacles:
            obstacle.draw(screen)
            obstacle.update()
            if player.dino_rect.colliderect(obstacle.rect):
                pygame.time.delay(0)
                death_count += 1
                menu(death_count)

        try:
            obstacle_distance = obstacles[-1].rect.x
            obstacle_width = obstacles[-1].rect.width
            obstacle_height = obstacles[-1].rect.height
        except:
            pass

        

        def score():
            global points, game_speed
            points += 1
            if points % 100 == 0:
                game_speed +=1

            text = font.render("Points: " + str(points), True, (0,0,0))
            textRect = text.get_rect()
            textRect.center = (1000,40)
            screen.blit(text, textRect)

        def backgroundDraw():
            global x_pos_bg, y_pos_bg

            image_width = background.get_width()
            screen.blit(background, (x_pos_bg, y_pos_bg))
            screen.blit(background, (image_width + x_pos_bg, y_pos_bg))

            if x_pos_bg <= -image_width:
                screen.blit(background, (image_width + x_pos_bg, y_pos_bg))
                x_pos_bg = 0
            x_pos_bg -= game_speed
        
        backgroundDraw()
        cloud.draw(screen)
        cloud.update()
        score()        
        clock.tick(30)
        pygame.display.update()

        def menu(death_count):
            global points
            run = True
            while run:
                screen.fill((255,255,255))
                font = pygame.font.Font('freesansbold.ttf', 20)

                if death_count == 0:
                    text = font.render("Press any key to start", True, (0,0,0))
                elif death_count > 0:
                    text = font.render("Press any key to restart", True, (0,0,0))
                    score = font.render("Your score: " +str(points), True, (0,0,0))
                    scoreRect = score.get_rect()
                    scoreRect.center = (screenW //2, screenH // 2 + 50)
                    screen.blit(score, scoreRect)
                textRect = text.get_rect()
                textRect.center = (screenW // 2, screenH // 2)
                screen.blit(text, textRect)
                pygame.display.update()
                main()
                for event in pygame.event.get():
                    if event.type ==pygame.QUIT:
                        run = False
                    #if event.type == pygame.KEYDOWN:
                        #main()
                    else:
                        main()

    
main()