import sys
import pygame

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt

import pygame
import sys

pygame.font.init()  # Inicjalizacja modułu czcionki
app = QtWidgets.QApplication(sys.argv)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)

font = pygame.font.SysFont(None, 24)

class LayerWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout()

        # Input neurons
        self.input_neurons = QtWidgets.QSpinBox()
        self.input_neurons.setMinimum(1)
        self.input_neurons.setMaximum(100)
        self.input_neurons.setValue(5)  # Domyślna wartość
        layout.addWidget(QtWidgets.QLabel("Input neurons"))
        layout.addWidget(self.input_neurons)

        # Output neurons
        self.output_neurons = QtWidgets.QSpinBox()
        self.output_neurons.setMinimum(1)
        self.output_neurons.setMaximum(100)
        self.output_neurons.setValue(4)  # Domyślna wartość
        layout.addWidget(QtWidgets.QLabel("Output neurons"))
        layout.addWidget(self.output_neurons)

        # Activation function
        self.activation_function = QtWidgets.QComboBox()
        self.activation_function.addItems(["sigmoid", "relu", "softmax", "tanh"])
        layout.addWidget(QtWidgets.QLabel("Activation"))
        layout.addWidget(self.activation_function)

        self.setLayout(layout)

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
                        self.options_window = OptionsWindow()
                        self.options_window.show()
                    if button["text"] == "Reset gen":
                        return True
                    if button["text"] == "Quit":
                        quit()

class LayerWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout()

        # Input neurons
        self.input_neurons = QtWidgets.QSpinBox()
        self.input_neurons.setMinimum(1)
        self.input_neurons.setMaximum(100)
        self.input_neurons.setValue(5)  # Domyślna wartość
        layout.addWidget(QtWidgets.QLabel("Input neurons"))
        layout.addWidget(self.input_neurons)

        # Output neurons
        self.output_neurons = QtWidgets.QSpinBox()
        self.output_neurons.setMinimum(1)
        self.output_neurons.setMaximum(100)
        self.output_neurons.setValue(4)  # Domyślna wartość
        layout.addWidget(QtWidgets.QLabel("Output neurons"))
        layout.addWidget(self.output_neurons)

        # Activation function
        self.activation_function = QtWidgets.QComboBox()
        self.activation_function.addItems(["sigmoid", "relu", "softmax", "tanh"])
        layout.addWidget(QtWidgets.QLabel("Activation"))
        layout.addWidget(self.activation_function)

        self.setLayout(layout)


class NeuralNetworkConfigWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Neural Network Configuration')
        self.setGeometry(100, 100, 400, 300)

        self.layout = QtWidgets.QVBoxLayout()

        # Warstwa wejściowa
        self.input_layer = InputLayerWidget()  # Warstwa wejściowa
        self.layout.addWidget(self.input_layer)

        # Kontener na warstwy środkowe
        self.layers_container = QtWidgets.QVBoxLayout()
        self.layout.addLayout(self.layers_container)

        # Warstwa wyjściowa
        self.output_layer = OutputLayerWidget()  # Warstwa wyjściowa
        self.layout.addWidget(self.output_layer)

        # Przycisk dodawania nowej warstwy
        self.add_layer_button = QtWidgets.QPushButton("Add Layer")
        self.add_layer_button.clicked.connect(self.add_layer)
        self.layout.addWidget(self.add_layer_button)

        # Przycisk zapisu
        self.save_button = QtWidgets.QPushButton("Save Configuration")
        self.save_button.clicked.connect(self.save_configuration)
        self.layout.addWidget(self.save_button)

        self.setLayout(self.layout)
        

    def add_layer(self):
        layer = LayerWidget()
        self.layers_container.addWidget(layer)

    def save_configuration(self):
        layers = []

        # Dodaj warstwę wejściową
        layers.append({
            "input_dim": 5,  # Stała wartość
            "output_dim": self.input_layer.output_neurons.value(),  # 3 stałe
            "activation": self.input_layer.activation_function.currentText()
        })

        for i in range(self.layers_container.count()):
            layer_widget = self.layers_container.itemAt(i).widget()
            layer = {
                "input_dim": layer_widget.input_neurons.value(),
                "output_dim": layer_widget.output_neurons.value(),
                "activation": layer_widget.activation_function.currentText()
            }
            layers.append(layer)

        # Dodaj warstwę wyjściową
        layers.append({
            "input_dim": self.output_layer.input_neurons.value(),  # Ostatnia warstwa
            "output_dim": 3,  # Stała wartość
            "activation": self.output_layer.activation_function.currentText()
        })

        # Zapisz do pliku w formacie wymaganym przez NN_ARCHITECTURE
        with open('layers_config.txt', 'w') as file:
            for layer in layers:
                file.write(f"input_dim: {layer['input_dim']}, output_dim: {layer['output_dim']}, activation: {layer['activation']}\n")

        print("Configuration saved to layers_config.txt")
        
 
class InputLayerWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QHBoxLayout()

        # Input Neurons (5 neurons, non-editable)
        self.input_neurons = QtWidgets.QSpinBox(self)
        self.input_neurons.setValue(5)  # Fixed value
        self.input_neurons.setReadOnly(True)  # Make it read-only
        layout.addWidget(QtWidgets.QLabel("Input Neurons"))
        layout.addWidget(self.input_neurons)

        # Output Neurons (3 neurons)
        self.output_neurons = QtWidgets.QSpinBox(self)
        self.output_neurons.setMinimum(1)  # Allow to change
        self.output_neurons.setMaximum(1000)
        self.output_neurons.setValue(3)  # Default value
        layout.addWidget(QtWidgets.QLabel("Output Neurons"))
        layout.addWidget(self.output_neurons)

        # Activation Function
        self.activation_function = QtWidgets.QComboBox(self)
        self.activation_function.addItems(["relu", "sigmoid", "softmax", "tanh"])
        layout.addWidget(QtWidgets.QLabel("Activation Function"))
        layout.addWidget(self.activation_function)

        self.setLayout(layout)

class OutputLayerWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QHBoxLayout()

        # Input Neurons (5 neurons)
        self.input_neurons = QtWidgets.QSpinBox(self)
        self.input_neurons.setMinimum(1)
        self.input_neurons.setMaximum(1000)
        self.input_neurons.setValue(3)
        layout.addWidget(QtWidgets.QLabel("Input Neurons"))
        layout.addWidget(self.input_neurons)

        # Output Neurons (3 neurons, non-editable)
        self.output_neurons = QtWidgets.QSpinBox(self)
        self.output_neurons.setValue(3)  # Fixed value
        self.output_neurons.setReadOnly(True)  # Make it read-only
        layout.addWidget(QtWidgets.QLabel("Output Neurons"))
        layout.addWidget(self.output_neurons)

        # Activation Function
        self.activation_function = QtWidgets.QComboBox(self)
        self.activation_function.addItems(["relu", "sigmoid", "softmax", "tanh"])
        layout.addWidget(QtWidgets.QLabel("Activation Function"))
        layout.addWidget(self.activation_function)

        self.setLayout(layout)

class LayerWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QHBoxLayout()

        # Input Neurons
        self.input_neurons = QtWidgets.QSpinBox(self)
        self.input_neurons.setMinimum(1)
        self.input_neurons.setMaximum(1000)
        layout.addWidget(QtWidgets.QLabel("Input Neurons"))
        layout.addWidget(self.input_neurons)

        # Output Neurons
        self.output_neurons = QtWidgets.QSpinBox(self)
        self.output_neurons.setMinimum(1)
        self.output_neurons.setMaximum(1000)
        layout.addWidget(QtWidgets.QLabel("Output Neurons"))
        layout.addWidget(self.output_neurons)

        # Activation Function
        self.activation_function = QtWidgets.QComboBox(self)
        self.activation_function.addItems(["relu", "sigmoid", "softmax", "tanh"])
        layout.addWidget(QtWidgets.QLabel("Activation Function"))
        layout.addWidget(self.activation_function)

        self.setLayout(layout)

class OptionsWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Options')
        self.setGeometry(100, 100, 600, 300)

        layout = QtWidgets.QVBoxLayout()

        # Game speed
        self.game_speed_field = QtWidgets.QDoubleSpinBox(self)
        self.game_speed_field.setMinimum(10)
        self.game_speed_field.setMaximum(100)
        self.game_speed_field.setSingleStep(1)
        layout.addWidget(QtWidgets.QLabel("Game Speed"))
        layout.addWidget(self.game_speed_field)

        # Jump height
        self.jump_height = QtWidgets.QDoubleSpinBox(self)
        self.jump_height.setMinimum(5)
        self.jump_height.setMaximum(15)
        layout.addWidget(QtWidgets.QLabel("Jump Height"))
        layout.addWidget(self.jump_height)

        # Bat check
        self.obstacle_bat = QtWidgets.QCheckBox(self)
        self.obstacle_bat.setCheckState(Qt.Checked)
        layout.addWidget(QtWidgets.QLabel("Spawn Bats"))
        layout.addWidget(self.obstacle_bat)

        # High obstacles check
        self.obstacle_high = QtWidgets.QCheckBox(self)
        self.obstacle_high.setCheckState(Qt.Checked)
        layout.addWidget(QtWidgets.QLabel("Spawn High Obstacles"))
        layout.addWidget(self.obstacle_high)

        # Short obstacles check
        self.obstacle_short = QtWidgets.QCheckBox(self)
        self.obstacle_short.setCheckState(Qt.Checked)
        layout.addWidget(QtWidgets.QLabel("Spawn Short Obstacles"))
        layout.addWidget(self.obstacle_short)

        # NN Settings Button
        self.nn_settings_button = QtWidgets.QPushButton("NN Settings", self)
        self.nn_settings_button.clicked.connect(self.open_nn_settings_window)  # Połączenie zdarzenia
        layout.addWidget(self.nn_settings_button)

        # Save Button
        save_button = QtWidgets.QPushButton("Apply", self)
        save_button.clicked.connect(self.save_settings_to_file)
        layout.addWidget(save_button)

        # Set layout
        self.setLayout(layout)
        
        self.nn_settings_window = None

    def open_nn_settings_window(self):
        if self.nn_settings_window is None:
            self.nn_settings_window = NeuralNetworkConfigWindow()  # Tworzenie instancji okna NN
        self.nn_settings_window.show()

    def save_settings_to_file(self):
        # Odczytanie wartości z kontrolek
        game_speed = self.game_speed_field.value()
        jump_height_value = self.jump_height.value()
        spawn_bats = self.obstacle_bat.isChecked()
        spawn_high_obs = self.obstacle_high.isChecked()
        spawn_short_obs = self.obstacle_short.isChecked()

        # Zapisz do pliku
        with open('settings.txt', 'w') as file:
            file.write(f"Game Speed: {game_speed}\n")
            file.write(f"Jump Height: {jump_height_value}\n")
            file.write(f"Spawn Bats: {spawn_bats}\n")
            file.write(f"Spawn High Obstacles: {spawn_high_obs}\n")
            file.write(f"Spawn Short Obstacles: {spawn_short_obs}\n")
        print(f"Settings saved to settings.txt")

    
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



