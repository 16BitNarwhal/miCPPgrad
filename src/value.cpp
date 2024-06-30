#include "value.h"
#include <vector>
#include <string>
#include <unordered_set>

#include <iostream>

Value::Value(double data, const std::unordered_set<Value*>& prev, const std::string& op)
	: data(data), grad(0.0), prev(prev), op(op) {}

Value Value::operator+(Value& other) {
	Value out{this->data + other.data, {this, &other}, "+"};
	out.backward = [&out, this, &other]() {
		this->grad += out.grad;
		other.grad += out.grad;
	};
	return out;
}

Value Value::operator-() {
	Value out{-this->data, {this}, "-"};
        out.backward = [&out, this]() {
                this->grad -= out.grad;
        };
        return out;;
}

Value Value::operator-(Value& other) {
        Value out{this->data - other.data, {this, &other}, "-"};
        out.backward = [&out, this, &other]() {
                this->grad += out.grad;
                other.grad -= out.grad;
        };
        return out;
}

Value Value::operator*(Value& other) {
        Value out{this->data * other.data, {this, &other}, "*"};
        out.backward = [&out, this, &other]() {
                this->grad += other.data * out.grad;
                other.grad += this->data * out.grad;
        };
        return out;
}
