import sys
import pygame

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt

import pygame
import sys

pygame.font.init()  # Inicjalizacja modułu czcionki


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)

font = pygame.font.SysFont(None, 24)

class Toolbar:
    def __init__(self, screen, width, height=50):
        self.screen = screen
        self.height = height
        self.width = width
        self.buttons = []
        self.button_size = (80, 40)
        self.button_margin = 10

    def add_button(self, text):
        if self.buttons:
            last_button = self.buttons[-1]
            pos = (last_button["pos"][0] + self.button_size[0] + self.button_margin, 5)
        else:
            pos = (10, 5)
        self.buttons.append({"text": text, "pos": pos})

    def draw(self):
        
        for button in self.buttons:
            self.draw_button(button["text"], button["pos"])

    def draw_button(self, text, pos):
        pygame.draw.rect(self.screen, GRAY, (*pos, *self.button_size))
        text_surface = font.render(text, True, BLACK)
        text_rect = text_surface.get_rect(center=(pos[0] + self.button_size[0] // 2, pos[1] + self.button_size[1] // 2))
        self.screen.blit(text_surface, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos
            for button in self.buttons:
                button_rect = pygame.Rect(button["pos"], self.button_size)
                if button_rect.collidepoint(mouse_pos):
                    print(f'Button {button["text"]} clicked')
                    if button["text"] == "Options":
                        open_options_window()
                    if button["text"] == "Quit":
                        quit()
                    if button["text"] == "Settings":
                        save_settings(game_speed_field, jump_height, duck_pos, obstacle_count, obstacle_bat, obstacle_high, obstacle_short)

def open_options_window():
    app = QtWidgets.QApplication(sys.argv)
    options_window = QtWidgets.QWidget()
    options_window.setWindowTitle('Options')
    options_window.setGeometry(100, 100, 600, 300)

    layout = QtWidgets.QVBoxLayout()

    #game speed
    label = QtWidgets.QLabel("Game Speed")
    game_speed_field = QtWidgets.QDoubleSpinBox(options_window)
    game_speed_field.setMinimum(10)
    game_speed_field.setMaximum(100)
    game_speed_field.setSingleStep(1)
    layout.addWidget(label)
    layout.addWidget(game_speed_field)
    

    #jump height
    label = QtWidgets.QLabel("Jump height")
    jump_height = QtWidgets.QDoubleSpinBox(options_window)
    jump_height.setMinimum(5)
    jump_height.setMaximum(15)
    layout.addWidget(label)
    layout.addWidget(jump_height)
    


    #duck pos
    # label = QtWidgets.QLabel("Duck positions")
    # duck_pos = QtWidgets.QSpinBox(options_window)
    # duck_pos.setMinimum(50)
    # duck_pos.setMaximum(500)
    # layout.addWidget(label)
    # layout.addWidget(duck_pos)


    # #obstacle count
    # label = QtWidgets.QLabel('Obstacle count')
    # obstacle_count = QtWidgets.QSlider(Qt.Horizontal, options_window)
    # obstacle_count.setMinimum(0)
    # obstacle_count.setMaximum(20)
    # obstacle_count.setSingleStep(1)
    # obstacle_count.valueChanged.connect(slider_logic)
    # obs_value = QtWidgets.QLabel('0')
    # layout.addWidget(label)
    # layout.addWidget(obstacle_count)
    # layout.addWidget(obs_value)

    #bat check
    label = QtWidgets.QLabel("Spawn bats")
    obstacle_bat = QtWidgets.QCheckBox(options_window)
    obstacle_bat.setCheckState(Qt.Checked)
    layout.addWidget(label)
    layout.addWidget(obstacle_bat)

    #high obs check
    label = QtWidgets.QLabel("Spawn high obstacles")
    obstacle_high = QtWidgets.QCheckBox(options_window)
    obstacle_high.setCheckState(Qt.Checked)
    layout.addWidget(label)
    layout.addWidget(obstacle_high)

    #short obs check
    label = QtWidgets.QLabel("Spawn short obstacles")
    obstacle_short = QtWidgets.QCheckBox(options_window)
    obstacle_short.setCheckState(Qt.Checked)
    layout.addWidget(label)
    layout.addWidget(obstacle_short)

    # Przycisk Zapisz
    save_button = QtWidgets.QPushButton("Apply", options_window)
    save_button.clicked.connect(lambda: save_settings_to_file('settings.txt',game_speed_field, jump_height, obstacle_bat, obstacle_high, obstacle_short))
    layout.addWidget(save_button)
    

    options_window.setLayout(layout)
    options_window.show()
    app.exec_()

def save_settings_to_file(filename, game_speed_field, jump_height, obstacle_bat, obstacle_high, obstacle_short):
    game_speed = game_speed_field.value()
    jump_height_value = jump_height.value()
    spawn_bats = obstacle_bat.isChecked()
    spawn_high_obs = obstacle_high.isChecked()
    spawn_short_obs = obstacle_short.isChecked()
    
    with open(filename, 'w') as file:
        file.write(f"Game Speed: {game_speed}\n")
        file.write(f"Jump Height: {jump_height_value}\n")
        file.write(f"Spawn Bats: {spawn_bats}\n")
        file.write(f"Spawn High Obstacles: {spawn_high_obs}\n")
        file.write(f"Spawn Short Obstacles: {spawn_short_obs}\n")
    print(f"Settings saved to {filename}")
    
def load_settings_from_file(filename):
    settings = {}
    try:
        with open(filename, 'r') as file:
            for line in file:
                key, value = line.strip().split(': ')
                settings[key] = value
        print(f"Settings loaded from {filename}")
    except FileNotFoundError:
        print(f"No settings file found: {filename}")
    
    return settings
    
def quit():
    pygame.quit()

def slider_logic(i):
    print(i)



