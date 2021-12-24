import random
import math

generation_size = 50
sample_size     = 5 

starting_weight_range = [-2, 2]
starting_bias_range   = [-4, 4]

mutation_percentage = 0.001 



def weight_generator(inputs, outputs):
    weights = []
    for i in range(outputs):
        output_associated_weights = []
        for j in range(inputs):
            output_associated_weights.append(random.uniform(starting_weight_range[0], starting_weight_range[1]))
        weights.append(output_associated_weights)
    return weights


class network_layer:
    def __init__(self, neurons, connections, model=None):
        self.neurons = [0] * neurons
        self.outputs = [0] * connections

        if model == None: self.weights = weight_generator(neurons, connections)
        else: self.weights = model
        


class neural_network:
    def __init__(self, architecture, model=None):
        self.fitness = 0
        self.layers = []

        if model == None:
            for i in range(len(architecture)-1): self.layers.append(network_layer(architecture[i], architecture[i+1]))
        else:
            for i in range(len(architecture)-1): self.layers.append(network_layer(architecture[i], architecture[i+1], model.layers[i].weights))


class evolutionary_model:
    def __init__(self):
        self.generation_sample = [] ##The best of the previous generation
        self.current_generation =[]

        self.architecture = [5,32,16,3]

    ##Randomly generates networks
    def start(self):
        for i in range(generation_size):
            self.current_generation.append(neural_network(self.architecture))

    ##Uses previously sampled networks as a base.
    def inherit(self):
        self.current_generation.clear()
        for i in range(int(generation_size/sample_size)):
            pass

    ##Sorts and samples the best performing networks.
    def sample(self):
        self.current_generation.sort(key = lambda gen: gen.fitness)
        self.generation_sample.clear()
        for i in range(sample_size): self.generation_sample.append(reversed(self.current_generation)[i])



if __name__ == "__main__":
    EM = evolutionary_model()
