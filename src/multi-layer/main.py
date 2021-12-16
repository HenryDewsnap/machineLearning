import matplotlib
import random
import json
import numpy as np

####    FUNCTIONS    ####
def sigmoid(x): return 1/(1+np.exp(-x))
#This value x must have been run through the sigmoid function first!
def sigmoid_gradient(x): return x * (1 - x)

def generate_weights(quantity, value_range):
    output = []
    for i in range(quantity): output.append(random.uniform(value_range[0], value_range[1]))
    return output

def matrix_maths(A, B, operator):
    output = []
    if len(A) == len(B):
        for ic in range(len(A)): output.append(eval(f"{A[ic]} {operator} {B[ic]}"))
    return output       
#### END OF FUNCTIONS ####



class layer:
    def __init__(self, neuron_count, forward_count):
        self.forward_count = forward_count
        self.neuron_count  = neuron_count

        self.neurons = [0] * neuron_count
        self.outputs = [0] * forward_count
        self.weights = []
        for i in range(forward_count): self.weights.append(generate_weights(neuron_count, [-1, 1]))

    def compile_outputs(self):
        output_neurons = [0] * self.forward_count
        for forward_connection in range(self.forward_count):
            for layer_neuron in range(self.neuron_count):
                output_neurons[forward_connection] += self.neurons[layer_neuron] * self.weights[forward_connection][layer_neuron]
        self.outputs.clear()
        for output in output_neurons: self.outputs.append(sigmoid(output))

    def update_inputs(self, inputs): self.neurons = inputs

    def return_output(self): return self.outputs

class network:
    def __init__(self, architecture):
        self.training_data = {}
        self.architecture = architecture
        self.network_layers = []
        for layer_ic in range(len(architecture)-1):
            self.network_layers.append(layer(architecture[layer_ic], architecture[layer_ic+1]))

    def feed_forward(self, inputs):
        previous_output = inputs
        for layer_ic in range(len(self.network_layers)):
            self.network_layers[layer_ic].update_inputs(previous_output)
            self.network_layers[layer_ic].compile_outputs()
            previous_output = self.network_layers[layer_ic].return_output()
        return previous_output

    def back_propagation(self, iterations):
        for iteration in range(iterations):
            print(f"Iteration {iteration}")
            for training_set in self.training_data:
                current_set = self.training_data[training_set]

                self.feed_forward(current_set['input']) ##Initialises each layer with output data by feeding an input through the network.
                
                training_outputs = current_set['output']
                training_inputs  = current_set['input']

                for layer in reversed(self.network_layers[1::]):
                    error = matrix_maths(training_outputs, layer.outputs, "-")
                    inputs = layer.neurons
                    outputs= layer.outputs

                    adjustments = []
                    for output_ic in range(len(outputs)): adjustments.append(outputs[output_ic] * error[output_ic])

                    for forward_connection in range(layer.forward_count):
                        for neuron_connection in range(layer.neuron_count):
                            layer.weights[forward_connection][neuron_connection] += inputs[neuron_connection] * adjustments[forward_connection]
                    
                    new_outputs = []
                    for neuron_ic in range(len(layer.neurons)):
                        new_outputs.append(0)
                        for output_ic in range(len(training_outputs)):
                            new_outputs[neuron_ic] += layer.weights[output_ic][neuron_ic] * training_outputs[output_ic]
                    
                    for i in range(len(new_outputs)):
                        new_outputs[i] = sigmoid(new_outputs[i])

                    training_outputs = new_outputs

    def load_training_data(self, path):
        self.training_data = json.load(open(path, "r"))


if __name__ == "__main__":
    NN = network([4,8,8,2])
    NN.load_training_data("training_data.json")
    NN.back_propagation(1000)
    
    t = NN.feed_forward([1,0,0,0])
    print(t)

    t = NN.feed_forward([0,0,1,1])
    print(t)
