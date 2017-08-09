#!/usr/bin/env python3

# Use this after running mnist_python2_dump.py
# This takes the cross-compatible dict, and turns it back into numpy
# 2d arrays, then serializes it using numpy's native zipped format.
# Should be run with python 3.

import pickle
import numpy as np
from json import dumps, loads

data = {}

with open('mnist.json', 'r') as input:
    data = loads(input.read())

# convert back to numpy data
training_x = np.array(data["training_x"], dtype="float32")
training_y = np.array(data["training_y"])

validation_x = np.array(data["validation_x"], dtype="float32")
validation_y = np.array(data["validation_y"])

test_x = np.array(data["test_x"], dtype="float32")
test_y = np.array(data["test_y"])

np.savez_compressed("mnist.npz", training_x, training_y, validation_x, validation_y, test_x, test_y)


