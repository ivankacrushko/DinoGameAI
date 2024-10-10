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