import json
import math
import random

class MLP:
    """
    A simple Multi-Layer Perceptron (Neural Network) implemented from scratch.
    """
    def __init__(self, layer_sizes: list[int]):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        
        # Initialize weights with He initialization, biases with zero
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i+1]
            
            limit = math.sqrt(2.0 / in_size)
            
            # w[out][in]
            w = [[random.uniform(-limit, limit) for _ in range(in_size)] for _ in range(out_size)]
            b = [0.0] * out_size
            
            self.weights.append(w)
            self.biases.append(b)
            
    def _relu(self, x: float) -> float:
        return x if x > 0 else 0.0
        
    def _relu_deriv(self, x: float) -> float:
        return 1.0 if x > 0 else 0.0
        
    def _softmax(self, z: list[float]) -> list[float]:
        max_z = max(z)
        exp_z = [math.exp(val - max_z) for val in z]
        sum_exp_z = sum(exp_z)
        return [val / sum_exp_z for val in exp_z]
        
    def forward(self, x: list[float]) -> tuple[list[float], list[list[float]], list[list[float]]]:
        """
        Forward pass.
        Returns output probabilities, activations (a), and pre-activations (z).
        """
        activations = [x]
        zs = []
        
        a = x
        for i in range(len(self.weights)):
            w = self.weights[i]
            b = self.biases[i]
            
            z = []
            for j in range(len(w)): # for each output node
                s = b[j]
                w_row = w[j]
                for k in range(len(a)):
                    s += w_row[k] * a[k]
                z.append(s)
            zs.append(z)
            
            if i == len(self.weights) - 1:
                # Output layer uses Softmax
                a = self._softmax(z)
            else:
                # Hidden layers use ReLU
                a = [self._relu(val) for val in z]
                
            activations.append(a)
            
        return a, activations, zs
        
    def train_step(self, x: list[float], y: int, learning_rate: float):
        """
        Performs a single gradient descent step for one sample.
        y is the index of the correct class.
        """
        preds, activations, zs = self.forward(x)
        
        # Calculate gradients for output layer (Softmax + CrossEntropy)
        delta = list(preds)
        delta[y] -= 1.0 # delta = p - y
        
        # Backpropagate
        for i in range(len(self.weights) - 1, -1, -1):
            w = self.weights[i]
            b = self.biases[i]
            a_prev = activations[i]
            
            # Gradients for weights and biases
            # dW[j][k] = delta[j] * a_prev[k]
            # db[j] = delta[j]
            
            if i > 0:
                # Calculate delta for previous layer
                z_prev = zs[i-1]
                delta_prev = [0.0] * len(z_prev)
                
                for k in range(len(z_prev)):
                    # sum over j of w[j][k] * delta[j]
                    s = 0.0
                    for j in range(len(w)):
                        s += w[j][k] * delta[j]
                    delta_prev[k] = s * self._relu_deriv(z_prev[k])
            
            # Apply gradients
            for j in range(len(w)):
                b[j] -= learning_rate * delta[j]
                for k in range(len(a_prev)):
                    w[j][k] -= learning_rate * delta[j] * a_prev[k]
                    
            if i > 0:
                delta = delta_prev
                
    def predict(self, x: list[float]) -> int:
        preds, _, _ = self.forward(x)
        return preds.index(max(preds))
        
    def save(self, path: str, classes: list[str], version: str = "1.0"):
        data = {
            "version": version,
            "architecture": self.layer_sizes,
            "classes": classes,
            "weights": self.weights,
            "biases": self.biases
        }
        with open(path, 'w') as f:
            json.dump(data, f)
            
    @classmethod
    def load(cls, path: str) -> tuple['MLP', list[str]]:
        with open(path, 'r') as f:
            data = json.load(f)
            
        model = cls(data["architecture"])
        model.weights = data["weights"]
        model.biases = data["biases"]
        
        return model, data["classes"]
