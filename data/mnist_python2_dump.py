#!/usr/bin/env python2

# This script takes the pickled data from python 2 in the file
# 'mnist.pkl.gz', and converts it into a dict of normal python lists.
# This allows it to be saved in a format which is completely cross-compatible with python 3.
# Run this with python 2.

import pickle
import gzip
from json import dumps, loads

f = gzip.open('mnist.pkl.gz', 'rb')
training, validation, test = pickle.load(f)
f.close()

training_x = training[0].tolist()
training_y = training[1].tolist()

validation_x = validation[0].tolist()
validation_y = validation[1].tolist()

test_x = test[0].tolist()
test_y = test[1].tolist()

data = {
    "training_x" : training_x,
    "training_y" : training_y,
    "validation_x" : validation_x,
    "validation_y" : validation_y,
    "test_x" : test_x,
    "test_y" : test_y
}

with open('mnist.json', 'w') as output:
    output.write(dumps(data))
