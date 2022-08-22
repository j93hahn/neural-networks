#!/bin/bash

python skip.py -i n -e one -freeze conv
python skip.py -i u -e one -freeze conv
python skip.py -i xn -e one -freeze conv
python skip.py -i xu -e one -freeze conv
python skip.py -i ku -fan in -e one -freeze conv
python skip.py -i ku -fan out -e one -freeze conv


python skip.py -i n -e ten -freeze conv
python skip.py -i u -e ten -freeze conv
python skip.py -i xn -e ten -freeze conv
python skip.py -i xu -e ten -freeze conv
python skip.py -i ku -fan in -e ten -freeze conv
python skip.py -i ku -fan out -e ten -freeze conv


python skip.py -i n -e one -freeze linear
python skip.py -i u -e one -freeze linear
python skip.py -i xn -e one -freeze linear
python skip.py -i xu -e one -freeze linear
python skip.py -i ku -fan in -e one -freeze linear
python skip.py -i ku -fan out -e one -freeze linear


python skip.py -i n -e ten -freeze linear
python skip.py -i u -e ten -freeze linear
python skip.py -i xn -e ten -freeze linear
python skip.py -i xu -e ten -freeze linear
python skip.py -i ku -fan in -e ten -freeze linear
python skip.py -i ku -fan out -e ten -freeze linear


python skip.py -i n -e one -freeze none
python skip.py -i u -e one -freeze none
python skip.py -i xn -e one -freeze none
python skip.py -i xu -e one -freeze none
python skip.py -i ku -fan in -e one -freeze none
python skip.py -i ku -fan out -e one -freeze none


python skip.py -i n -e ten -freeze none
python skip.py -i u -e ten -freeze none
python skip.py -i xn -e ten -freeze none
python skip.py -i xu -e ten -freeze none
python skip.py -i ku -fan in -e ten -freeze none
python skip.py -i ku -fan out -e ten -freeze none