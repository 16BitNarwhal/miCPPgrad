#ifndef VALUE_H
#define VALUE_H

#include <unordered_set>
#include <string>
#include <functional>

class Value {
public:
	double data;
	double grad;
	std::unordered_set<Value*> prev;
	std::function<void()> backward;
	std::string op;

	Value(double data, const std::unordered_set<Value*>& prev = {}, const std::string& op = "");

	Value operator+(Value& other);
};

#endif
