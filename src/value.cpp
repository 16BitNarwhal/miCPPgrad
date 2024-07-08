#include "value.h"

#include <iostream>

Value::Value(double data, const std::unordered_set<std::shared_ptr<Value>>& prev, const std::string& op)
    : data(data), grad(0.0), prev(prev), op(op) {}

void Value::backprop() {
    std::vector<std::shared_ptr<Value>> topo;
    std::unordered_set<std::shared_ptr<Value>> visited;

    std::function<void(const std::shared_ptr<Value>&)> build_topo = [&](const std::shared_ptr<Value>& v) {
        if (visited.find(v) != visited.end()) return;
        visited.insert(v);
        for (const std::shared_ptr<Value>& prev : v->prev) {
            build_topo(prev);
        }
        topo.push_back(v);
    };

    build_topo(shared_from_this());

    grad = 1.0;

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->backward != nullptr) {
            (*it)->backward();
        }
    }
}

std::shared_ptr<Value> Value::operator+(const std::shared_ptr<Value>& other) {
    std::shared_ptr<Value> out = std::make_shared<Value>(Value(this->data + other->data, {shared_from_this(), other}, "+"));
    std::weak_ptr<Value> weak_out = out;
    out->backward = [weak_out, this, other]() {
        if (std::shared_ptr<Value> out = weak_out.lock()) {
            this->grad += out->grad;
            other->grad += out->grad;
        }
    };
    return out;
}

std::shared_ptr<Value> Value::operator-() {
    return *this * std::make_shared<Value>(-1.0);
}

std::shared_ptr<Value> Value::operator-(const std::shared_ptr<Value>& other) {
    return *this + (-other);
}

std::shared_ptr<Value> Value::operator*(const std::shared_ptr<Value>& other) {
    std::shared_ptr<Value> out = std::make_shared<Value>(Value(this->data * other->data, {shared_from_this(), other}, "*"));
    std::weak_ptr<Value> weak_out = out;
    out->backward = [weak_out, this, other]() {
        if (std::shared_ptr<Value> out = weak_out.lock()) {
            this->grad += other->data * out->grad;
            other->grad += this->data * out->grad;
        }
    };
    return out;
}

std::shared_ptr<Value> Value::operator/(const std::shared_ptr<Value>& other) {
    return *this * other->pow(std::make_shared<Value>(-1.0));
}

std::shared_ptr<Value> Value::pow(const std::shared_ptr<Value>& other) {
    std::shared_ptr<Value> out = std::make_shared<Value>(Value(std::pow(this->data, other->data), {shared_from_this(), other}, "^"));
    std::weak_ptr<Value> weak_out = out;
    out->backward = [weak_out, this, other]() {
        if (std::shared_ptr<Value> out = weak_out.lock()) {
            this->grad += other->data * std::pow(this->data, other->data-1) * out->grad;
            other->grad += std::log(this->data) * out->data * out->grad;
        }
    };
    return out;
}

std::shared_ptr<Value> Value::tanh() {
    std::shared_ptr<Value> x = std::make_shared<Value>(log(2));
    std::shared_ptr<Value> ex = std::make_shared<Value>(exp(1))->pow(std::make_shared<Value>(2) * x);
    std::shared_ptr<Value> out = (ex - std::make_shared<Value>(1)) / (ex + std::make_shared<Value>(1));
    return out;
}

std::shared_ptr<Value> Value::relu() {
    std::shared_ptr<Value> out = std::make_shared<Value>(Value(std::max(0.0, this->data), {shared_from_this()}, "relu"));
    std::weak_ptr<Value> weak_out = out;
    out->backward = [weak_out, this]() {
        if (std::shared_ptr<Value> out = weak_out.lock()) {
            this->grad += (out->data > 0) * out->grad;
        }
    };
    return out;
}

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return (*a) + b;
}

std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a) {
    return -(*a);
}

std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return (*a) - b;
}

std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return (*a) * b;
}

std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return (*a) / b;
}

std::shared_ptr<Value> pow(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return (*a).pow(b);
}