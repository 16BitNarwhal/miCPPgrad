#ifndef VALUE_H
#define VALUE_H

#include <unordered_set>
#include <string>
#include <functional>
#include <cmath>
#include <memory>

class Value : public std::enable_shared_from_this<Value> {
public:
	double data;
	double grad;
	std::unordered_set<std::shared_ptr<Value>> prev;
	std::function<void()> backward;
	std::string op;

	Value(double data, const std::unordered_set<std::shared_ptr<Value>>& prev = {}, const std::string& op = "");

	std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& other);
	std::shared_ptr<Value> operator-();
	std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& other);
	std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& other);
	std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& other);
	std::shared_ptr<Value> pow(const std::shared_ptr<Value>& other);
};

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b);
std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a);
std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b);
std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b);
std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b);
std::shared_ptr<Value> pow(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b);

#endif
