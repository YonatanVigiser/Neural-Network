import random
import numpy as np
from tqdm import tqdm


class NeuralNetwork:
    def __init__(self, neurons_count: list[int]):
        self.neurons_count = neurons_count
        self.hidden_layers_count = len(neurons_count) - 2

        self.weights = self.generate_random_weights()
        self.biases = self.generate_random_biases()

    @staticmethod
    def sig(x: float) -> float:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def cost(desired_values: list[float], values: list[float]) -> float:
        res = 0
        for i in range(len(desired_values)):
            res += (values[i] - desired_values[i]) ** 2
        return res

    # Impacts:
    @staticmethod
    def impact_of_output_weight(activation: float, desired_activation: float, activation_before: float) -> float:
        return 2 * (activation - desired_activation) * activation * (1 - activation) * activation_before

    @staticmethod
    def impact_of_output_bias(activation: float, desired_value: float) -> float:
        return 2 * (activation - desired_value) * activation * (1 - activation)

    @staticmethod
    def impact_of_output_activation(activation: float, desired_values: list[float], weights: list[float]) -> float:
        return sum([2 * (activation - desired_values[i]) * activation * (1 - activation) *
                    weights[i] for i in range(len(weights))])

    @staticmethod
    def impact_of_weight(activation: float, activation_before: float, layer_after_impact: float) -> float:
        return activation * (1 - activation) * activation_before * layer_after_impact

    @staticmethod
    def impact_of_bias(activation: float, this_layer_impact: float) -> float:
        return activation * (1 - activation) * this_layer_impact

    @staticmethod
    def impact_of_activation(activation: float, weights: list[float], next_layer_impact: list[float]) -> float:
        return sum([activation * (1 - activation) * weights[i] * next_layer_impact[i] for i in range(len(weights))])

    def weights_impact_one_layer(self, network_activations: list[list[float]], layer: int, desired_output: list[float],
                                 next_layer_impact: list[float]) -> list[list[float]]:
        layer_activation = network_activations[layer]
        layer_before_activation = network_activations[layer - 1]

        res = [[] for _ in range(len(layer_activation))]

        if len(next_layer_impact) == 0:
            for i in range(len(layer_activation)):
                for x in range(len(layer_before_activation)):
                    res[i].append(round(self.impact_of_output_weight(layer_activation[i], desired_output[i],
                                                                     layer_before_activation[x]), 15))
        else:
            for i in range(len(layer_activation)):
                for x in range(len(layer_before_activation)):
                    res[i].append(
                        round(self.impact_of_weight(layer_activation[i], layer_before_activation[x],
                                                    next_layer_impact[i]), 15))

        return res

    def bias_impact_one_layer(self, network_activations: list[list[float]], layer: int, desired_output: list[float],
                              this_layer_impact: list[float]) -> list[float]:
        layer_activation = network_activations[layer]
        res = []

        if len(this_layer_impact) == 0:
            for i in range(len(layer_activation)):
                res.append(self.impact_of_output_bias(layer_activation[i], desired_output[i]))

        else:
            for i in range(len(layer_activation)):
                res.append(self.impact_of_bias(layer_activation[i], this_layer_impact[i]))

        return res

    def activation_impact_one_layer(self, network_activations: list[list[float]], layer: int,
                                    desired_outputs: list[float], normal_weights, next_layer_impact):
        layer_activation = network_activations[layer - 1]
        res = []
        weights = [[] for _ in range(len(layer_activation))]

        for i in range(len(normal_weights[layer - 1])):
            for x in range(len(normal_weights[layer - 1][i])):
                weights[x].append(normal_weights[layer - 1][i][x])

        if len(next_layer_impact) == 0:
            for i in range(len(layer_activation)):
                res.append(round(self.impact_of_output_activation(
                    layer_activation[i], desired_outputs, weights[i]), 15))
        else:
            for i in range(len(layer_activation)):
                res.append(round(self.impact_of_activation(layer_activation[i], weights[i], next_layer_impact), 15))

        return res

    def generate_random_weights(self) -> list[list[list[float]]]:
        weights = [[] for _ in range(self.hidden_layers_count + 1)]

        for i in range(1, len(self.neurons_count)):
            for x in range(self.neurons_count[i]):
                weights[i - 1].append([round(random.uniform(-1, 1), 10) for _ in range(self.neurons_count[i - 1])])

        return weights

    def generate_random_biases(self) -> list[list[float]]:
        biases = [[] for _ in range(self.hidden_layers_count + 1)]

        for i in range(1, len(self.neurons_count)):
            for _ in range(self.neurons_count[i]):
                biases[i - 1].append(round(random.uniform(-0.5, 0.5), 10))

        return biases

    def generate_biases_zero(self) -> list[list[float]]:
        biases = [[] for _ in range(self.hidden_layers_count + 1)]

        for i in range(1, len(self.neurons_count)):
            for _ in range(self.neurons_count[i]):
                biases[i - 1].append(0)
        return biases

    def run_one_layer(self, layer_num: int, inputs: list[float]) -> list[float]:
        output = []
        weights = self.weights[layer_num]
        biases = self.biases[layer_num]

        for i in range(len(biases)):
            res = biases[i]
            for x in range(len(weights[i])):
                res += weights[i][x] * inputs[x]
            res = float(self.sig(res))
            output.append(res)

        return output

    def run_network(self, inputs: list[float]) -> list[list[float]]:
        result = [inputs]  # added inputs directly to history
        for i in range(len(self.weights)):
            layer = self.run_one_layer(i, inputs)
            inputs = layer

            result.append(layer)

        return result

    def train(self, training_inputs: list[list[float]], training_outputs: list[int], learning_rate: float) -> None:
        print("Training started!")
        for i in tqdm(range(len(training_outputs))):
            inputs = training_inputs[i]
            desired_output = training_outputs[i]
            network = self.run_network(inputs)
            # network.insert(0, inputs)

            impact = [[] for i in range(self.hidden_layers_count + 3)]
            desired_values = [0] * self.neurons_count[-1]
            desired_values[desired_output] = 1

            for layer in range(self.hidden_layers_count + 1, 0, -1):
                weights_impact = self.weights_impact_one_layer(network, layer, desired_values, impact[layer + 1])
                impact[layer] = self.activation_impact_one_layer(network, layer, desired_values,
                                                                 self.weights, impact[layer + 1])

                if layer == self.hidden_layers_count + 1:
                    biases_impact = self.bias_impact_one_layer(network, layer, desired_values, [])
                else:
                    biases_impact = self.bias_impact_one_layer(network, layer, desired_values, impact[layer])

                for n in range(len(weights_impact)):
                    self.biases[layer - 1][n] = self.biases[layer - 1][n] - biases_impact[n] * learning_rate
                    for x in range(len(weights_impact[n])):
                        self.weights[layer - 1][n][x] = (self.weights[layer - 1][n][x] - weights_impact[n][x] *
                                                         learning_rate)
        print("\nFinished Training!\n")

    def test(self, training_inputs: list[list[float]], training_outputs: list[int]) -> float:
        print("Test started!")
        correct = 0
        for i in tqdm(range(len(training_outputs))):
            network = self.run_network(training_inputs[i])
            prediction = network[-1].index(max(network[-1]))
            if prediction == training_outputs[i]:
                correct += 1

        percentage = (100 * correct) / len(training_outputs)
        print(f"\nFinished testing with {percentage}% success rate!")
        return percentage
