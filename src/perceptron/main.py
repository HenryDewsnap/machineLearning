import random
import math
import json


randFloat = lambda l, h: random.uniform(l,h)

def sigmoid(x): return 1/(1+math.exp(-x))

#This value x must have been run through the sigmoid function first!
def sigmoid_gradient(x): return x * (1 - x)


def Error(desired, actual): return desired-actual

def findNeuronValue(inputs, weights):
    output = 0
    if len(inputs) == len(weights):
        for i in range(len(inputs)):
            output += inputs[i] * weights[i]
    return output

def loadJson(filePath):
    try: return json.load(open(filePath, "r"))
    except: exit(f"File {filePath} Not Found")

##Stages:
##1.) Inputs are taken from Xn input neurons.
##2.) Inputs are multiplied by their Wn weighted values.
##3.) The weighted values are summed at the neuron and then run through an activation function (in this case sigmoid).
##4.) The activated function data is then forwarded to the output Y.
##5.) The output is then error checked and the slight imperfections are corrected, and then this process is re-iterated until the dataset
##    Is exausted or until it is interrupted.

## weight adjust = error * input * gradient of sigmoid(output)
## We use the gradient of the sigmoid function because however great the gradient is dictates the networks confidence in its answer, deciding
## Whether we make a large change to our weighted value or a small change.

class neuralNetwork:
    def __init__(self, neurons, iterations, trainingData):
        self.inputNeurons = [0] * neurons
        self.synapticWeights = []
        self.trainingData = loadJson(trainingData)
        self.iterations = iterations

        for i in range(neurons):
            self.synapticWeights.append(randFloat(0,1))

    def trainNet(self):
        for Iteration in range(int(self.iterations/len(self.trainingData))):
            for subIteration in self.trainingData:
                print(f"Currently Training: {subIteration}")

                currentData = self.trainingData[subIteration]
                desired_output = currentData['output']
                input_layer = currentData['input']
                print(f"{input_layer}, {self.synapticWeights}")
                output = sigmoid(findNeuronValue(input_layer, self.synapticWeights))

                error = Error(desired_output[0], output) ##I had to address 0, because the data was a list.

                adjustments = error * sigmoid_gradient(output)

                for weight_pointer in range(len(self.synapticWeights)):
                    self.synapticWeights[weight_pointer] += input_layer[weight_pointer] * adjustments
                
    def test(self, input, doDesired=False):
        output = sigmoid(findNeuronValue(input, self.synapticWeights))
        print(f"\nInputs: {input} || Network Output: {output}")

        if doDesired == True:
            foundDesired = False
            for data in self.trainingData:
                if self.trainingData[data]['input'] == input:
                    foundDesired = True
                    desired = self.trainingData[data]['output']
                    break
        if foundDesired == True: 
            print(f"\nDesired Output: {desired}")
            print(f"Error: {Error(desired[0], output)} \n")
        else: print("Desired value could not be found.")

if __name__ == "__main__":
    network = neuralNetwork(3,10000,"training_data.json")
    network.trainNet()
    network.test([1,1,1], True) ##This one is hard particuarlaly for the network to do.
