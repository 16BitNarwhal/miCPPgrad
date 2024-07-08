#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "value.h"
#include <vector>
#include <random>
#include <iostream>

class Module {
public:
    void zero_grad();

    virtual std::vector<std::shared_ptr<Value>> get_parameters() = 0;
};

class Neuron : public Module {
public:
    std::vector<std::shared_ptr<Value>> weights;
    std::shared_ptr<Value> bias;
    std::shared_ptr<Value> output;

    Neuron(const int n_inputs);

    std::shared_ptr<Value> operator()(const std::vector<std::shared_ptr<Value>>& inputs);
    std::vector<std::shared_ptr<Value>> get_parameters() override;

    friend std::ostream& operator<<(std::ostream& os, const Neuron& n);
};

class Layer : public Module {
public:
    std::vector<Neuron> neurons;

    Layer(const int n_inputs, const int n_outputs);

    std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& inputs);
    std::vector<std::shared_ptr<Value>> get_parameters() override;

    friend std::ostream& operator<<(std::ostream& os, const Layer& l);
};

class MultiLayerPerceptron : public Module {
public:
    std::vector<Layer> layers;

    MultiLayerPerceptron(const std::vector<int>& layer_sizes);

    std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& inputs);
    std::vector<std::shared_ptr<Value>> get_parameters() override;

    friend std::ostream& operator<<(std::ostream& os, const MultiLayerPerceptron& mlp);
};

#endif
