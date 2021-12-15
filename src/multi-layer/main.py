import random
import math
import json

##architecture example:
##[5,3,3,2]
##5 input neurons.
##3 hidden neurons.
##3 hidden neurons.
##2 output neurons.


def loadJson(filePath):
    try: return json.load(open(filePath, "r"))
    except: exit(f"File {filePath} Not Found")

def generate_weights(count):
    array = []
    for c in range(count): array.append(random.uniform(0,1))
    return array

def sigmoid(x): return 1/(1+math.exp(-x))
def sigmoid_gradient(x): return x * (1 - x)

class layer:
    def __init__(self, neuron_count, forward_connections):
        self.forward_connections = forward_connections
        self.neurons = [0]*neuron_count
        self.synaptic_weights = []
        for fc in range(self.forward_connections): self.synaptic_weights.append(generate_weights(len(self.neurons)))

    def calculate_output(self, inputs):
        self.neurons = inputs
        outputs = []
        for forward_connection in range(self.forward_connections):
            outputs.append(0)
            for neuron_ic in range(len(self.neurons)):
                outputs[forward_connection] += self.neurons[neuron_ic] * self.synaptic_weights[forward_connection][neuron_ic]
        for output_ic in range(len(outputs)): outputs[output_ic] = sigmoid(outputs[output_ic])
        return outputs

    def train_layer(self, desired_outputs, output):
        for forward_connection in range(self.forward_connections):
            error = desired_outputs[forward_connection] - output[forward_connection]
            adjustments = error * sigmoid_gradient(output[forward_connection])

            for neuron_ic in range(len(self.neurons)):
                self.synaptic_weights[forward_connection][neuron_ic] += self.neurons[neuron_ic] * adjustments


    def display_layer(self): print(f"\nInfo:\n- Layer Neurons:{len(self.neurons)}\n- Inheriting Neurons: {self.forward_connections}\n")



class neuralNet:
    def __init__(self, architecture, iterations, trainingData):
        self.layers = []
        for layer_ic in range(len(architecture)-1):
            self.layers.append(layer(architecture[layer_ic], architecture[layer_ic+1]))


    def execute_layers(self, inputs):
        previous_layer = inputs
        print(f"Inputs: {inputs}")
        for layer in self.layers:
            previous_layer = layer.calculate_output(previous_layer)
        print(f"Outputs: {previous_layer}")



if __name__ == "__main__":
    a = neuralNet([5,3,3,2], 10000, "training_data.json")
    a.execute_layers([0,1,0,1,0])
