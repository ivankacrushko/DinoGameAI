
import pygame
import os
import random
import numpy as np
###
from keyboard import press, release
import pyautogui
import time
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from scipy import stats



pygame.init()

screenH = 600
screenW = 1100
screen = pygame.display.set_mode((screenW, screenH))


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
    X_POS = 80
    Y_POS = 310
    Y_POS_DUCK = 340
    JUMP_VEL = 8.5

    def __init__(self):
        self.duck_img = ducking
        self.run_img = running
        self.jump_img = jumping

        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False
        self.is_jumping = False

        self.step_index = 0
        self.jump_vel = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

    def update(self, userInput, nnInput):
        if self.dino_duck:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()

        if self.step_index >= 10:
            self.step_index = 0

        if (userInput[pygame.K_UP] or nnInput[2]) and not self.dino_jump:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
            self.is_jumping = True
            self.jump_vel = self.JUMP_VEL
        elif (userInput[pygame.K_DOWN] or nnInput[1]) and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
        elif not (self.dino_jump or userInput[pygame.K_DOWN]):
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False

    def jump(self):
        self.image = self.jump_img
        if self.dino_jump:
            self.dino_rect.y -= self.jump_vel * 4
            self.jump_vel -= 0.8
        if self.dino_rect.y >= self.Y_POS:
            self.dino_jump = False            
            self.jump_vel = self.JUMP_VEL

    def duck(self):
        self.image = self.duck_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS_DUCK
        self.step_index += 1

    def run(self):
        self.image = self.run_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.step_index += 1

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))

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



class Model():
    def __init__(self):
        self.params_values = 0

    def init_layers(self, nn_architecture, seed = 99):
        np.random.seed(seed)
        number_of_layers = len(nn_architecture)
        params_values = {}
    
        for idx, layer in enumerate(nn_architecture):
            layer_idx = idx + 1
        
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]
        
            params_values['W' + str(layer_idx)] = np.random.randn(
                layer_output_size, layer_input_size) * 0.1
            params_values['b' + str(layer_idx)] = np.random.randn(
                layer_output_size, 1) * 0.1
            self.params_values = params_values
        return params_values

    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def softmax_backward(self, softmax_values, target_labels):
        return softmax_values - target_labels

    def sigmoid_backward(self, dA, Z):
        sig = self.sigmoid(Z)
        return dA * sig * (1 - sig)


    def single_layer_forward_propagation(self, A_prev, W_curr, b_curr, activation="sigmoid"):
        Z_curr = np.dot(W_curr, A_prev) + b_curr
        if activation is "softmax":
            activation_func = self.softmax
        elif activation is "sigmoid":
            activation_func = self.sigmoid
        else:
            raise Exception('error')
    
        return activation_func(Z_curr), Z_curr

    def full_forward_propagation(self, X, params_values, nn_architecture):
    
        memory = {}
        A_curr = X
        for idx, layer in enumerate(nn_architecture):
            layer_idx = idx + 1
            A_prev = A_curr
        
            activ_function_curr = layer["activation"]
            W_curr = params_values["W" + str(layer_idx)]
            b_curr = params_values["b" + str(layer_idx)]
            A_curr, Z_curr = self.single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)
        
            memory["A" + str(idx)] = A_prev
            memory["Z" + str(layer_idx)] = Z_curr
        
        return A_curr, memory

    def get_cost_value(self, Y_hat, Y):
        epsilon = 1e-15  
        Y_hat = np.clip(Y_hat, epsilon, 1 - epsilon)
        cross_entropy = - np.sum(Y * np.log(Y_hat))
        cross_entropy /= len(Y)

        return cross_entropy

    def convert_prob_into_class(self, probs):
        probs_ = np.copy(probs)
        probs_[probs_ > 0.5] = 1
        probs_[probs_ <= 0.5] = 0
        return probs_

    def get_accuracy_value(self, Y_hat, Y):
        Y_hat_ = self.convert_prob_into_class(Y_hat)
        return (Y_hat_ == Y).all(axis=0).mean()

    def single_layer_backward_propagation(self, dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="sigmoid"):
        m = A_prev.shape[1]
        if activation is "softmax":
            backward_activation_func = self.softmax_backward
        elif activation is "sigmoid":
            backward_activation_func = self.sigmoid_backward
        else:
            raise Exception('error')
    
        dZ_curr = backward_activation_func(dA_curr, Z_curr)
    
        dW_curr = np.dot(dZ_curr, A_prev.T) / m
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr

    def full_backward_propagation(self, Y_hat, Y, memory, params_values, nn_architecture):
        grads_values = {}
        m = Y.shape[1]
        Y = Y.reshape(Y_hat.shape)
        dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat));
    
        for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
            layer_idx_curr = layer_idx_prev + 1
            activ_function_curr = layer["activation"]
        
            dA_curr = dA_prev
        
            A_prev = memory["A" + str(layer_idx_prev)]
            Z_curr = memory["Z" + str(layer_idx_curr)]
        
            W_curr = params_values["W" + str(layer_idx_curr)]
            b_curr = params_values["b" + str(layer_idx_curr)]
        
            dA_prev, dW_curr, db_curr = self.single_layer_backward_propagation(
                dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
        
            grads_values["dW" + str(layer_idx_curr)] = dW_curr
            grads_values["db" + str(layer_idx_curr)] = db_curr
    
        return grads_values

    def update(self, params_values, grads_values, nn_architecture, learning_rate):
        for layer_idx, layer in enumerate(nn_architecture, 1):
            params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]        
            params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

        return params_values;

    def train(self, X, Y, nn_architecture, epochs, learning_rate, verbose=False, callback=None):
        params_values = self.init_layers(nn_architecture, 99)
        cost_history = []
        accuracy_history = []
    
        for i in range(epochs):
            Y_hat, cashe = self.full_forward_propagation(X, params_values, nn_architecture)   
            cost = self.get_cost_value(Y_hat, Y)
            cost_history.append(cost)
            accuracy = self.get_accuracy_value(Y_hat, Y)
            accuracy_history.append(accuracy)
            grads_values = self.full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)
            params_values = self.update(params_values, grads_values, nn_architecture, learning_rate)
        
            if(i % 50 == 0):
                if(verbose):
                    print("Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f}".format(i, cost, accuracy))
                if(callback is not None):
                    callback(i, params_values)
            
        return params_values

NN_ARCHITECTURE = [
        {"input_dim": 5, "output_dim": 4, "activation": "sigmoid"},
        {"input_dim": 4, "output_dim": 3, "activation": "softmax"},
    ] 

gen1 = Model()
gen1.train(np.transpose(train_X), np.transpose(train_Y), NN_ARCHITECTURE, 1000, 0.0009)



def main():
    global model, game_speed, x_pos_bg, y_pos_bg,highest_score, points,obstacles, obstacle_distance, obstacle_width, obstacle_height, death_count
    run = True
    clock = pygame.time.Clock()
    player = Dinosaur()    
    cloud = Cloud()
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
        
        text2 = font.render("Smierci: " + str(death_count), True, (0,0,0))
        textRect = text2.get_rect()
        textRect.center = (1000,80)
        screen.blit(text2, textRect)

    def backgroundDraw():
        global x_pos_bg, y_pos_bg

        image_width = background.get_width()
        screen.blit(background, (x_pos_bg, y_pos_bg))
        screen.blit(background, (image_width + x_pos_bg, y_pos_bg))

        if x_pos_bg <= -image_width:
            screen.blit(background, (image_width + x_pos_bg, y_pos_bg))
            x_pos_bg = 0
        x_pos_bg -= game_speed    
    

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        screen.fill((255,255,255))
        userInput = pygame.key.get_pressed()
        
        
        
        ####
        inputs = np.array([[obstacle_distance, obstacle_height, obstacle_width, game_speed, is_flying]])    

        player.draw(screen)
        gen1_forward, rr = gen1.full_forward_propagation(np.reshape(inputs, (5,1)), gen1.params_values, NN_ARCHITECTURE)
        player.update(userInput, filter_actions(gen1_forward))

        ####

        if len(obstacles) == 0:
            if random.randint(0,2) == 0:
                obstacles.append(SmallCactus(smallCactus))
                is_flying = -1
            elif random.randint(0,2) == 1:
                obstacles.append(LargeCactus(largeCactus))
                is_flying = -1
            elif random.randint(0,2) == 2:
                obstacles.append(Bird(bird))
                is_flying = 1

        for obstacle in obstacles:
            obstacle.draw(screen)
            obstacle.update()
            if player.dino_rect.colliderect(obstacle.rect):
                death_count += 1
                menu(death_count)

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
        cloud.update()
        score(highest_score, death_count)        
        clock.tick(30)
        pygame.display.update()

def menu(death_count):
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
        pygame.display.update()
        for event in pygame.event.get():
            if event.type ==pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                main()
                    

    
menu(death_count)