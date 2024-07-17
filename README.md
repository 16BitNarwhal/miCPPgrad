# miCPPgrad
miCPPgrad (my c++ grad) is a tiny scalar-valued autograd engine in c++ inspired by [Kaparthy's micrograd]([url](https://github.com/karpathy/micrograd))

in `src/main.cpp`, i wrote a sample MLP for 3D surface regression task given some noisily generated points from a ground_truth function

you can train it with
```shell
./run.sh
```

run testcases for the Value engine
```shell
./run.sh test
```

skip the build by adding `skipbuild` as an arg
```shell
./run.sh skipbuild
```

after running `./run.sh`, you can generate a visualization gif of the training process with `python graph.py`

please make sure you have `matplotlib` and `pandas` installed in your environment to run the graphing script

here's a sample i made, trained model surface on the left and ground truth on right with noisy data on both sides
![](https://github.com/16BitNarwhal/miCPPgrad/blob/dev/neural_network_3d_graph_demo.gif?raw=true)
