#!/bin/bash

python cnn-experiment.py -m lenet -i z -n bn -c i --numeric --summary
python cnn-experiment.py -m vgg -i z -n bn -c i --numeric --summary
python cnn-experiment.py -m lenet -i o -n bn -c i --numeric --summary
python cnn-experiment.py -m vgg -i o -n bn -c i --numeric --summary
python cnn-experiment.py -m lenet -i n -n bn -c i --numeric --summary
python cnn-experiment.py -m vgg -i n -n bn -c i --numeric --summary
python cnn-experiment.py -m lenet -i u -n bn -c i --numeric --summary
python cnn-experiment.py -m vgg -i u -n bn -c i --numeric --summary
python cnn-experiment.py -m lenet -i xn -n bn -c i --numeric --summary
python cnn-experiment.py -m vgg -i xn -n bn -c i --numeric --summary
python cnn-experiment.py -m lenet -i xu -n bn -c i --numeric --summary
python cnn-experiment.py -m vgg -i xu -n bn -c i --numeric --summary


python cnn-experiment.py -m lenet -i ku -n bn -c n -f in --numeric --summary
python cnn-experiment.py -m lenet -i ku -n ln -c n -f in --numeric --summary
python cnn-experiment.py -m lenet -i ku -n gn -c n -f in --numeric --summary
python cnn-experiment.py -m lenet -i ku -n nn -c n -f in --numeric --summary
python cnn-experiment.py -m vgg -i ku -n bn -c n -f in --numeric --summary
python cnn-experiment.py -m vgg -i ku -n ln -c n -f in --numeric --summary
python cnn-experiment.py -m vgg -i ku -n gn -c n -f in --numeric --summary
python cnn-experiment.py -m vgg -i ku -n nn -c n -f in --numeric --summary


python cnn-experiment.py -m lenet -i n -n bn -c n --numeric --summary
python cnn-experiment.py -m lenet -i n -n ln -c n --numeric --summary
python cnn-experiment.py -m lenet -i n -n gn -c n --numeric --summary
python cnn-experiment.py -m lenet -i n -n nn -c n --numeric --summary
python cnn-experiment.py -m vgg -i n -n bn -c n --numeric --summary
python cnn-experiment.py -m vgg -i n -n ln -c n --numeric --summary
python cnn-experiment.py -m vgg -i n -n gn -c n --numeric --summary
python cnn-experiment.py -m vgg -i n -n nn -c n --numeric --summary


python cnn-experiment.py -m lenet -i o -n nn -c n --numeric --summary
python cnn-experiment.py -m vgg -i o -n nn -c n --numeric --summary


python cnn-experiment.py -m lenet -i u -n bn -c n --summary
python cnn-experiment.py -m lenet -i u -n ln -c n --summary
python cnn-experiment.py -m lenet -i u -n gn -c n --summary
python cnn-experiment.py -m lenet -i u -n nn -c n --summary
python cnn-experiment.py -m vgg -i u -n bn -c n --summary
python cnn-experiment.py -m vgg -i u -n ln -c n --summary
python cnn-experiment.py -m vgg -i u -n gn -c n --summary
python cnn-experiment.py -m vgg -i u -n nn -c n --summary


python cnn-experiment.py -m lenet -i xu -n bn -c n --summary
python cnn-experiment.py -m lenet -i xu -n ln -c n --summary
python cnn-experiment.py -m lenet -i xu -n gn -c n --summary
python cnn-experiment.py -m lenet -i xu -n nn -c n --summary
python cnn-experiment.py -m vgg -i xu -n bn -c n --summary
python cnn-experiment.py -m vgg -i xu -n ln -c n --summary
python cnn-experiment.py -m vgg -i xu -n gn -c n --summary
python cnn-experiment.py -m vgg -i xu -n nn -c n --summary


python cnn-experiment.py -m lenet -i xn -n bn -c n --summary
python cnn-experiment.py -m lenet -i xn -n ln -c n --summary
python cnn-experiment.py -m lenet -i xn -n gn -c n --summary
python cnn-experiment.py -m lenet -i xn -n nn -c n --summary
python cnn-experiment.py -m vgg -i xn -n bn -c n --summary
python cnn-experiment.py -m vgg -i xn -n ln -c n --summary
python cnn-experiment.py -m vgg -i xn -n gn -c n --summary
python cnn-experiment.py -m vgg -i xn -n nn -c n --summary


python cnn-experiment.py -m lenet -i ku -n bn -c n -f out --summary
python cnn-experiment.py -m lenet -i ku -n ln -c n -f out --summary
python cnn-experiment.py -m lenet -i ku -n gn -c n -f out --summary
python cnn-experiment.py -m lenet -i ku -n nn -c n -f out --summary
python cnn-experiment.py -m vgg -i ku -n bn -c n -f out --summary
python cnn-experiment.py -m vgg -i ku -n ln -c n -f out --summary
python cnn-experiment.py -m vgg -i ku -n gn -c n -f out --summary
python cnn-experiment.py -m vgg -i ku -n nn -c n -f out --summary