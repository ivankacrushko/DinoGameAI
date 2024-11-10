import sys
import pygame

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt

import pygame
import sys

import loading

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

        #self.load_configuration()
        

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
                
    def load_configuration(self):
    # Clear existing widgets in the container
        while self.layers_container.count() > 0:
            item = self.layers_container.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        try:
            with open('layers_config.txt', 'r') as file:
                lines = file.readlines()
                for line in lines:
                    # Parse each line to extract layer properties
                    params = line.strip().split(',')
                    layer_params = {}
                    for param in params:
                        key, value = param.split(': ')
                        layer_params[key.strip()] = value.strip()

                    # Create a new LayerWidget and set its properties
                    layer_widget = LayerWidget()
                    layer_widget.input_neurons.setValue(int(layer_params['input_dim']))
                    layer_widget.output_neurons.setValue(int(layer_params['output_dim']))
                    layer_widget.activation_function.setCurrentText(layer_params['activation'])
                
                    # Add the configured layer to the container
                    self.layers_container.addWidget(layer_widget)
                
        except FileNotFoundError:
            print("Configuration file not found.")



        
 
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

class TrainingConfigWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.learning_rate_value, self.epochs_value, method = loading.load_training_settings("neuron_training_config.txt")        

        self.setWindowTitle('Training Configuration')
        self.setGeometry(100, 100, 200, 100)

        self.layout = QtWidgets.QVBoxLayout()

        learning_rate_layout = QtWidgets.QHBoxLayout()
        learning_rate_layout.setSpacing(5)
        self.learning_rate = QtWidgets.QDoubleSpinBox(self)
        self.learning_rate.setMinimum(0.0001)
        self.learning_rate.setMaximum(1)
        self.learning_rate.setSingleStep(0.0001)
        self.learning_rate.setDecimals(4)
        self.learning_rate.setValue(self.learning_rate_value)
        learning_rate_layout.addWidget(QtWidgets.QLabel("Learning rate"))
        learning_rate_layout.addWidget(self.learning_rate)
        self.layout.addLayout(learning_rate_layout)
        
        
        epoch_layout = QtWidgets.QHBoxLayout()
        epoch_layout.setSpacing(5)
        self.epochs = QtWidgets.QDoubleSpinBox(self)
        self.epochs.setMinimum(1)
        self.epochs.setMaximum(30000)
        self.epochs.setSingleStep(100)
        self.epochs.setDecimals(0)
        self.epochs.setValue(self.epochs_value)
        epoch_layout.addWidget(QtWidgets.QLabel("Epochs"))
        epoch_layout.addWidget(self.epochs)
        self.layout.addLayout(epoch_layout)

        method_layout = QtWidgets.QHBoxLayout()
        method_layout.setSpacing(5)
        self.method = QtWidgets.QComboBox()
        self.method.addItem("Backpropagation")
        self.method.addItem("Genetic")
        self.method.setCurrentText(method)
        method_layout.addWidget(QtWidgets.QLabel("Training method:"))
        method_layout.addWidget(self.method)
        self.layout.addLayout(method_layout)
        
        

        # Przycisk zapisu
        self.save_button = QtWidgets.QPushButton("Save Configuration")
        self.save_button.clicked.connect(self.save_configuration)
        self.layout.addWidget(self.save_button)

        self.setLayout(self.layout)

    def save_configuration(self):
        learning_rate = self.learning_rate.value()
        epochs = self.epochs.value()
        method = self.method.currentText()
        with open('neuron_training_config.txt', 'w') as file:
            
            file.write(f"learning_rate: {learning_rate}, epochs: {epochs}, method: {method}")

       
        
        

class OptionsWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        
        game_speed, jump_height, spawn_bats, spawn_high, spawn_short = loading.load_game_settings("game_settings.txt")

        self.setWindowTitle('Options')
        self.setGeometry(100, 100, 600, 300)

        layout = QtWidgets.QVBoxLayout()

        # Game speed
        self.game_speed_field = QtWidgets.QDoubleSpinBox(self)
        self.game_speed_field.setMinimum(10)
        self.game_speed_field.setMaximum(100)
        self.game_speed_field.setSingleStep(1)
        self.game_speed_field.setValue(game_speed)
        layout.addWidget(QtWidgets.QLabel("Game Speed"))
        layout.addWidget(self.game_speed_field)

        # Jump height
        self.jump_height = QtWidgets.QDoubleSpinBox(self)
        self.jump_height.setMinimum(5)
        self.jump_height.setMaximum(15)
        self.jump_height.setSingleStep(0.1)
        self.jump_height.setValue(jump_height)
        layout.addWidget(QtWidgets.QLabel("Jump Height"))
        layout.addWidget(self.jump_height)

        # Bat check
        bat_checkbox_layout = QtWidgets.QHBoxLayout()        
        bat_checkbox_layout.setSpacing(5)
        self.obstacle_bat = QtWidgets.QCheckBox(self)
        if spawn_bats == True:
            self.obstacle_bat.setChecked(True)
        else:
            self.obstacle_bat.setChecked(False)
        bat_checkbox_layout.addWidget(self.obstacle_bat)
        bat_checkbox_layout.addWidget(QtWidgets.QLabel("Spawn bats"))
        bat_checkbox_layout.addStretch(1)
        layout.addLayout(bat_checkbox_layout)

        # High obstacles check
        high_obs_checkbox_layout = QtWidgets.QHBoxLayout()        
        high_obs_checkbox_layout.setSpacing(5)
        self.obstacle_high = QtWidgets.QCheckBox(self)
        if spawn_high == True:
            self.obstacle_high.setChecked(True)
        else:
            self.obstacle_high.setChecked(False)
        high_obs_checkbox_layout.addWidget(self.obstacle_high)
        high_obs_checkbox_layout.addWidget(QtWidgets.QLabel("Spawn high obstacles"))
        high_obs_checkbox_layout.addStretch(1)
        layout.addLayout(high_obs_checkbox_layout)

        # Short obstacles check
        short_obs_checkbox_layout = QtWidgets.QHBoxLayout()        
        short_obs_checkbox_layout.setSpacing(5)
        self.obstacle_short = QtWidgets.QCheckBox(self)
        if spawn_short == True:
            self.obstacle_short.setChecked(True)
        else:
            self.obstacle_short.setChecked(False)
        short_obs_checkbox_layout.addWidget(self.obstacle_short)
        short_obs_checkbox_layout.addWidget(QtWidgets.QLabel("Spawn short obstacles"))
        short_obs_checkbox_layout.addStretch(1)
        layout.addLayout(short_obs_checkbox_layout)

        # NN Settings Button
        self.nn_settings_button = QtWidgets.QPushButton("Neural network settings", self)
        self.nn_settings_button.clicked.connect(self.open_nn_settings_window)
        layout.addWidget(self.nn_settings_button)
        
        #Train Settings Button
        self.train_settings_button = QtWidgets.QPushButton("Neuron training settings", self)
        self.train_settings_button.clicked.connect(self.open_train_settings_window) 
        layout.addWidget(self.train_settings_button)

        # Save Button
        save_button = QtWidgets.QPushButton("Apply", self)
        save_button.clicked.connect(self.save_settings_to_file)
        layout.addWidget(save_button)

        # Set layout
        self.setLayout(layout)
        
        self.nn_settings_window = None
        self.train_settings_window = None

    def open_nn_settings_window(self):
        if self.nn_settings_window is None:
            self.nn_settings_window = NeuralNetworkConfigWindow()
        self.nn_settings_window.show()

    def open_train_settings_window(self):
        if self.train_settings_window is None:
            self.train_settings_window = TrainingConfigWindow()
        self.train_settings_window.show()

    def save_settings_to_file(self):
        # Odczytanie wartości z kontrolek
        game_speed = self.game_speed_field.value()
        jump_height_value = self.jump_height.value()
        spawn_bats = self.obstacle_bat.isChecked()
        spawn_high_obs = self.obstacle_high.isChecked()
        spawn_short_obs = self.obstacle_short.isChecked()

        # Zapisz do pliku
        with open('game_settings.txt', 'w') as file:
            file.write(f"game_speed: {game_speed}, ")
            file.write(f"jump_height: {jump_height_value}, ")
            file.write(f"spawn_bats: {spawn_bats}, ")
            file.write(f"spawn_high: {spawn_high_obs}, ")
            file.write(f"spawn_short: {spawn_short_obs}")
        
    
# def load_settings_from_file(filename):
#     settings = {}
#     try:
#         with open(filename, 'r') as file:
#             for line in file:
#                 key, value = line.strip().split(': ')
#                 settings[key] = value
#         print(f"Settings loaded from {filename}")
#     except FileNotFoundError:
#         print(f"No settings file found: {filename}")
    
#     return settings
    
def quit():
    pygame.quit()

def slider_logic(i):
    print(i)



