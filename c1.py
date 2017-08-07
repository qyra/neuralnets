import random
import numpy as np

class Network:
    def __init__(self, layer_neuron_counts):
        """ layer_neuron_counts
                An array representing how many neurons there are
                per layer in network.
                layer_neuron_counts[0] = numinputs
                layer_neuron_counts[end] = numoutputs
        """

        self.layer_count = len(layer_neuron_counts)
        self.sizes = sizes
        # Initialize from standard bell curve.
        self.biases = [np.random.randn(c, 1) for c in layer_neuron_counts[1:]]

    def dump(self):
        print(self.s)

n = Network([2,3,1])
n.dump()
