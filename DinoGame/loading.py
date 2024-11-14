def load_architecture_from_file(filename):
    architecture = []

    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Rozdziel ka¿dy wiersz wed³ug przecinków
            params = line.strip().split(',')
            layer = {}
            for param in params:
                key, value = param.split(': ')
                layer[key.strip()] = value.strip()  # Dodaj parametry do warstwy
            architecture.append(layer)  # Dodaj warstwê do architektury

    return architecture

  
def format_nn_architecture(layers):
    nn_architecture = []
    for layer in layers:
        nn_architecture.append({
            "input_dim": int(layer["input_dim"]),
            "output_dim": int(layer["output_dim"]),
            "activation": layer["activation"]
        })
    return nn_architecture

def load_training_settings(filename):
   with open(filename, 'r') as file:
        lines = file.readlines()
            
        for line in lines:
            params = line.strip().split(',')
            settings = {}
            for param in params:
                key, value = param.split(': ')
                settings[key.strip()] = value.strip()
                
        return format_training_settings(settings)

def format_training_settings(settings):    
    settings = {
        'method' : settings['method'],
        'learning_rate' : float(settings['learning_rate']),
        'epochs' : int(float(settings['epochs'])),
        'mutation_rate' : float(settings['mutation_rate']),
        'generations' : int(float(settings['generations'])),
        'population_size' : int(float(settings['population_size']))
        }

    return settings

def load_game_settings(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
            
        for line in lines:
            params = line.strip().split(',')
            settings = {}
            for param in params:
                key, value = param.split(': ')
                settings[key.strip()] = value.strip()
                
        return format_game_settings(settings)
        
def format_game_settings(settings):
    game_speed = float(settings["game_speed"])
    jump_height = float(settings['jump_height'])
    spawn_bats = eval(settings['spawn_bats'])
    spawn_high = eval(settings['spawn_high'])
    spawn_short = eval(settings['spawn_short'])
        
    return game_speed, jump_height, spawn_bats, spawn_high, spawn_short