import random
import numpy as np
from scipy.special import expit

class Network:
    def version(self):
        print("This is version 2 of c1")
        d = dependency.DepClass()
        d.version()

    def __init__(self, sizes):
        """ layer_neuron_counts
                An array representing how many neurons there are
                per layer in network.
                layer_neuron_counts[0] = numinputs
                layer_neuron_counts[end] = numoutputs
        """

        self.num_layers = len(sizes)
        self.sizes = sizes
        # Initialize from standard bell curve.
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def learn(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)

        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            # Split training data into batches of size batch_size
            batches = []
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def update_mini_batch(self, batch, eta):
        # Add gradients of all batches
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            delta_grad_b, delta_grad_w = self.backprop(x, y)
            grad_b = np.add(grad_b, delta_grad_b)
            grad_w = np.add(grad_w, delta_grad_w)

        scale = eta/len(batch)
        self.weights = [w-nw*scale for w, nw in zip(self.weights, grad_w)]
        self.biases = [b-nb*scale for b, nb in zip(self.biases, grad_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = expit(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = expit(np.dot(w, a) + b)

        return a

def sigmoid_prime(x):
    """Derivative of the sigmoid function."""
    return expit(x)*(1-expit(x))

if __name__ == "__main__":
    print("Testing network")
    net = Network([784, 30, 10])

    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    print("Training:")
    net.learn(training_data, 30, 10, 3.0, test_data=test_data)



