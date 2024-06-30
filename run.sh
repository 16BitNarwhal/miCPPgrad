#!/bin/sh

mkdir -p build
cd build
cmake ..
make

if [ "$1" = "test" ]
then
	./micppgrad_test
else
	./micppgrad
fi
