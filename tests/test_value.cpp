#include <gtest/gtest.h>
#include "value.h"
#include <cmath>
#include <memory>

#include <iostream>

using namespace std;

TEST(ValueTest, CreateValue) {
    Value a{5.0};
    ASSERT_EQ(a.data, 5.0);
}

TEST(ValueTest, Add2Values) {
    shared_ptr<Value> a = make_shared<Value>(5.0);
    shared_ptr<Value> b = make_shared<Value>(3.0);
    shared_ptr<Value> c = a + b;
    ASSERT_EQ(c->data, 8.0);
}

TEST(ValueTest, NegateValue) {
    shared_ptr<Value> a = make_shared<Value>(5.0);
    shared_ptr<Value> b = -a;
    ASSERT_EQ(b->data, -5.0);
}

TEST(ValueTest, Sub2Values) {
    shared_ptr<Value> a = make_shared<Value>(6.0);
    shared_ptr<Value> b = make_shared<Value>(3.5);
    shared_ptr<Value> c = a - b;

    ASSERT_EQ(c->data, 2.5);
}

TEST(ValueTest, Mult2Values) {
    shared_ptr<Value> a = make_shared<Value>(4.0);
    shared_ptr<Value> b = make_shared<Value>(7.0);
    shared_ptr<Value> c = a * b;
    ASSERT_EQ(c->data, 28.0);
}

TEST(ValueTest, Div2Values) {
    shared_ptr<Value> a = make_shared<Value>(12.0);
    shared_ptr<Value> b = make_shared<Value>(3.0);
    shared_ptr<Value> c = a / b;
    ASSERT_EQ(c->data, 4.0);
}

TEST(ValueTest, PowValue) {
    shared_ptr<Value> a = make_shared<Value>(2.0);
    shared_ptr<Value> b = make_shared<Value>(3.0);
    shared_ptr<Value> c = pow(a, b);
    ASSERT_EQ(c->data, 8.0);
}

TEST(ValueTest, Add2ValuesGrad) {
    shared_ptr<Value> a = make_shared<Value>(5.0);
    shared_ptr<Value> b = make_shared<Value>(3.0);
    shared_ptr<Value> c = a + b;
	
    c->backprop();

    ASSERT_EQ(a->grad, 1.0);
    ASSERT_EQ(b->grad, 1.0);
}

TEST(ValueTest, NegateValueGrad) {
    shared_ptr<Value> a = make_shared<Value>(5.0);
    shared_ptr<Value> b = -a;
	
    b->backprop();

    ASSERT_EQ(a->grad, -1.0);
}

TEST(ValueTest, Sub2ValuesGrad) {
    shared_ptr<Value> a = make_shared<Value>(6.0);
    shared_ptr<Value> b = make_shared<Value>(3.5);

    shared_ptr<Value> c = a - b;

    c->backprop();
        
    ASSERT_EQ(a->grad, 1.0);
    ASSERT_EQ(b->grad, -1.0);
}

TEST(ValueTest, Mult2ValuesGrad) {
    shared_ptr<Value> a = make_shared<Value>(4.0);
    shared_ptr<Value> b = make_shared<Value>(7.0);
    shared_ptr<Value> c = a * b;

    c->backprop();
    ASSERT_EQ(a->grad, 7.0);
    ASSERT_EQ(b->grad, 4.0);
}

TEST(ValueTest, Div2ValuesGrad) {
    shared_ptr<Value> a = make_shared<Value>(12.0);
    shared_ptr<Value> b = make_shared<Value>(3.0);
    shared_ptr<Value> c = a / b;
	
    c->backprop();
    ASSERT_NEAR(a->grad, 1.0/3.0, 1e-6);
    ASSERT_NEAR(b->grad, -4.0/3.0, 1e-6);
}

TEST(ValueTest, Pow2ValuesGrad) {
    shared_ptr<Value> a = make_shared<Value>(exp(1));
    shared_ptr<Value> b = make_shared<Value>(3.0);
    shared_ptr<Value> c = pow(a, b);
	
    c->backprop();

    ASSERT_NEAR(a->grad, 3*exp(2), 1e-6);
    ASSERT_NEAR(b->grad, exp(3), 1e-6);
}

TEST(ValueTest, ManualTanh) {
    shared_ptr<Value> x = make_shared<Value>(log(2));
    shared_ptr<Value> ex = pow(make_shared<Value>(exp(1)), make_shared<Value>(2) * x);
    shared_ptr<Value> result = (ex - make_shared<Value>(1)) / (ex + make_shared<Value>(1));

    ASSERT_NEAR(result->data, 3.0/5.0, 1e-6);

    result->backprop();

    ASSERT_NEAR(x->grad, 0.64, 1e-6);
}

TEST(ValueTest, ReluZero) {
    shared_ptr<Value> x = make_shared<Value>(-21.0);
    shared_ptr<Value> result = x->relu();

    ASSERT_EQ(result->data, 0.0);

    result->backprop();

    ASSERT_EQ(x->grad, 0.0);
}

TEST(ValueTest, ReluNonZero) {
    shared_ptr<Value> x = make_shared<Value>(14.0);
    shared_ptr<Value> result = x->relu();

    ASSERT_EQ(result->data, 14.0);

    result->backprop();

    ASSERT_EQ(x->grad, 1.0);
}