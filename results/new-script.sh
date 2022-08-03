#!/bin/bash

python cnn-experiment.py -m lenet -i u -n bn -c i --numeric --summary --loss
python cnn-experiment.py -m vgg -i u -n bn -c i --numeric --summary --loss
python cnn-experiment.py -m lenet -i xu -n bn -c i --numeric --summary --loss
python cnn-experiment.py -m vgg -i xu -n bn -c i --numeric --summary --loss


python cnn-experiment.py -m lenet -i ku -n bn -c n -f in --numeric --summary --loss
python cnn-experiment.py -m lenet -i ku -n ln -c n -f in --numeric --summary --loss
python cnn-experiment.py -m lenet -i ku -n gn -c n -f in --numeric --summary --loss
python cnn-experiment.py -m lenet -i ku -n nn -c n -f in --numeric --summary --loss
python cnn-experiment.py -m vgg -i ku -n bn -c n -f in --numeric --summary --loss
python cnn-experiment.py -m vgg -i ku -n ln -c n -f in --numeric --summary --loss
python cnn-experiment.py -m vgg -i ku -n gn -c n -f in --numeric --summary --loss
python cnn-experiment.py -m vgg -i ku -n nn -c n -f in --numeric --summary --loss


python cnn-experiment.py -m lenet -i u -n bn -c n --numeric --summary --loss
python cnn-experiment.py -m lenet -i u -n ln -c n --numeric --summary --loss
python cnn-experiment.py -m lenet -i u -n gn -c n --numeric --summary --loss
python cnn-experiment.py -m lenet -i u -n nn -c n --numeric --summary --loss
python cnn-experiment.py -m vgg -i u -n bn -c n --numeric --summary --loss
python cnn-experiment.py -m vgg -i u -n ln -c n --numeric --summary --loss
python cnn-experiment.py -m vgg -i u -n gn -c n --numeric --summary --loss
python cnn-experiment.py -m vgg -i u -n nn -c n --numeric --summary --loss


python cnn-experiment.py -m lenet -i xu -n bn -c n --numeric --summary --loss
python cnn-experiment.py -m lenet -i xu -n ln -c n --numeric --summary --loss
python cnn-experiment.py -m lenet -i xu -n gn -c n --numeric --summary --loss
python cnn-experiment.py -m lenet -i xu -n nn -c n --numeric --summary --loss
python cnn-experiment.py -m vgg -i xu -n bn -c n --numeric --summary --loss
python cnn-experiment.py -m vgg -i xu -n ln -c n --numeric --summary --loss
python cnn-experiment.py -m vgg -i xu -n gn -c n --numeric --summary --loss
python cnn-experiment.py -m vgg -i xu -n nn -c n --numeric --summary --loss


python cnn-experiment.py -m lenet -i ku -n bn -c n -f out --loss
python cnn-experiment.py -m lenet -i ku -n ln -c n -f out --loss
python cnn-experiment.py -m lenet -i ku -n gn -c n -f out --loss
python cnn-experiment.py -m lenet -i ku -n nn -c n -f out --loss
python cnn-experiment.py -m vgg -i ku -n bn -c n -f out --loss
python cnn-experiment.py -m vgg -i ku -n ln -c n -f out --loss
python cnn-experiment.py -m vgg -i ku -n gn -c n -f out --loss
python cnn-experiment.py -m vgg -i ku -n nn -c n -f out --loss