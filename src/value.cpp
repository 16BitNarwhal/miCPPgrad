#include "value.h"

#include <iostream>

Value::Value(double data, const std::unordered_set<std::shared_ptr<Value>>& prev, const std::string& op)
    : data(data), grad(0.0), prev(prev), op(op) {}

std::shared_ptr<Value> Value::operator+(const std::shared_ptr<Value>& other) {
    std::shared_ptr<Value> out = std::make_shared<Value>(Value(this->data + other->data, {shared_from_this(), other}, "+"));
    out->backward = [out, this, other]() {
        this->grad += out->grad;
        other->grad += out->grad;
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
    out->backward = [out, this, other]() {
        this->grad += other->data * out->grad;
        other->grad += this->data * out->grad;
    };
    return out;
}

std::shared_ptr<Value> Value::operator/(const std::shared_ptr<Value>& other) {
    return *this * other->pow(std::make_shared<Value>(-1.0));
}

std::shared_ptr<Value> Value::pow(const std::shared_ptr<Value>& other) {
    std::shared_ptr<Value> out = std::make_shared<Value>(Value(std::pow(this->data, other->data), {shared_from_this(), other}, "^"));
    out->backward = [out, this, other]() {
        this->grad += other->data * std::pow(this->data, other->data-1) * out->grad;
        other->grad += std::log(this->data) * out->data * out->grad;
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