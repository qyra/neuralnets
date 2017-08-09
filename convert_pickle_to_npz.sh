#!/bin/bash
# Converts mnist.pkl.gz (python2 version) into mnist.npz (python3 version)
cd data
python2 mnist_python2_dump.py
python3 mnist_python_3_serialize.py
rm mnist.json
