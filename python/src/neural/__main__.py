import matplotlib.pyplot as plt
import numpy as np
from .neural import NeuralNetwork

input_vectors = [
    [3, 1.5],
    [2, 1],
    [4, 1.5],
    [3, 4],
    [3.5, 0.5],
    [2, 0.5],
    [5.5, 1],
    [1, 1],
]
targets = np.array([0, 1, 0, 1, 0, 1, 1, 0])
learning_rate = 0.1

def main():
    neural_network = NeuralNetwork(learning_rate)
    training_error = neural_network.train(input_vectors, targets, 10000)
    plt.plot(training_error)
    plt.xlabel("Iterations")
    plt.ylabel("Error for all training instances")
    plt.savefig("cumulative_error.png")
    plt.show()

if __name__ == "__main__":
    main()
