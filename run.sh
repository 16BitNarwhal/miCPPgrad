#!/bin/sh

skip_build=false
run_tests=false

# Check arguments
for arg in "$@"; do
    if [ "$arg" = "test" ]; then
        run_tests=true
    fi
    if [ "$arg" = "skipbuild" ]; then
        skip_build=true
    fi
done

# Build if skip_build is not true
if [ "$skip_build" != true ]; then
    mkdir -p build
    cd build || exit 1  
    if ! (cmake .. && make); then
        exit 1
    fi
    cd .. 
fi

# Run appropriate binary based on arguments
if [ "$run_tests" = true ]; then
    ./build/micppgrad_test
else
    ./build/micppgrad
fi

