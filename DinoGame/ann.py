import numpy as np


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
        if activation == "softmax":
            activation_func = self.softmax
        elif activation == "sigmoid":
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
        if activation == "softmax":
            backward_activation_func = self.softmax_backward
        elif activation == "sigmoid":
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
        
            
        print("Epoka: {:05} trafnosc: {:.5f}".format(1000, accuracy))
                
            
        return params_values

NN_ARCHITECTURE = [
        {"input_dim": 5, "output_dim": 4, "activation": "sigmoid"},
        {"input_dim": 4, "output_dim": 3, "activation": "softmax"},
    ] 