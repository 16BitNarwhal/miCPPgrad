#include "neural_network.h"

void Module::zero_grad() {
    std::vector<std::shared_ptr<Value>> parameters = get_parameters();
    for (std::shared_ptr<Value> p : parameters) {
        p->grad = 0.0;
    }
}

Neuron::Neuron(const int n_inputs) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (int i = 0; i < n_inputs; i++) {
        this->weights.push_back(std::make_shared<Value>(Value(dis(gen))));
    }
    this->bias = std::make_shared<Value>(Value(dis(gen)));
}

std::shared_ptr<Value> Neuron::operator()(const std::vector<std::shared_ptr<Value>>& inputs) {
    std::shared_ptr<Value> out = std::make_shared<Value>(Value(0.0));
    for (int i = 0; i < inputs.size(); i++) {
        out->data += inputs[i]->data * this->weights[i]->data;
    }
    out->data += this->bias->data;
    return out;
}

std::vector<std::shared_ptr<Value>> Neuron::get_parameters() {
    std::vector<std::shared_ptr<Value>> parameters = {this->bias};
    for (std::shared_ptr<Value>& w : this->weights) {
        parameters.push_back(w);
    }
    return parameters;
}

Layer::Layer(const int n_inputs, const int n_neurons) {
    for (int i = 0; i < n_neurons; i++) {
        this->neurons.push_back(Neuron(n_inputs));
    }
}

std::vector<std::shared_ptr<Value>> Layer::operator()(const std::vector<std::shared_ptr<Value>>& inputs) {
    std::vector<std::shared_ptr<Value>> outputs;
    for (Neuron& neuron : this->neurons) {
        outputs.push_back(neuron(inputs));
    }
    return outputs;
}

std::vector<std::shared_ptr<Value>> Layer::get_parameters() {
    std::vector<std::shared_ptr<Value>> parameters;
    for (auto& n : this->neurons) {
        parameters.insert(parameters.end(), n.get_parameters().begin(), n.get_parameters().end());
    }
    return parameters;
}

MultiLayerPerceptron::MultiLayerPerceptron(const std::vector<int>& layer_sizes) {
    for (int i = 0; i < layer_sizes.size() - 1; i++) {
        this->layers.push_back(Layer(layer_sizes[i], layer_sizes[i + 1]));
    }
}

std::vector<std::shared_ptr<Value>> MultiLayerPerceptron::operator()(const std::vector<std::shared_ptr<Value>>& inputs) {
    std::vector<std::shared_ptr<Value>> outputs = inputs;
    for (Layer& layer : this->layers) {
        outputs = layer(outputs);
    }
    return outputs;
}

std::vector<std::shared_ptr<Value>> MultiLayerPerceptron::get_parameters() {
    std::vector<std::shared_ptr<Value>> parameters;
    for (auto& l : this->layers) {
        parameters.insert(parameters.end(), l.get_parameters().begin(), l.get_parameters().end());
    }
    return parameters;
}