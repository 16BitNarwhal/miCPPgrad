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
	
    c->grad = 2.0;
    c->backward();

    ASSERT_EQ(a->grad, 2.0);
    ASSERT_EQ(b->grad, 2.0);
}

TEST(ValueTest, NegateValueGrad) {
    shared_ptr<Value> a = make_shared<Value>(5.0);
    shared_ptr<Value> b = -a;
	
    b->grad = 2.0;
    b->backward();

    ASSERT_EQ(a->grad, -2.0);
}

TEST(ValueTest, Sub2ValuesGrad) {
    shared_ptr<Value> a = make_shared<Value>(6.0);
    shared_ptr<Value> b = make_shared<Value>(3.5);

    shared_ptr<Value> c = a - b;

    c->grad = 2.0;
    c->backward();
        
    ASSERT_EQ(a->grad, 2.0);
    ASSERT_EQ(b->grad, -2.0);
}

TEST(ValueTest, Mult2ValuesGrad) {
    shared_ptr<Value> a = make_shared<Value>(4.0);
    shared_ptr<Value> b = make_shared<Value>(7.0);
    shared_ptr<Value> c = a * b;

    c->grad = 2.0;
    c->backward();
    ASSERT_EQ(a->grad, 14.0);
    ASSERT_EQ(b->grad, 8.0);
}

TEST(ValueTest, Div2ValuesGrad) {
    shared_ptr<Value> a = make_shared<Value>(12.0);
    shared_ptr<Value> b = make_shared<Value>(3.0);
    shared_ptr<Value> c = a / b;
	
    c->grad = 2.0;
    c->backward();
    ASSERT_NEAR(a->grad, 2.0/3.0, 1e-6);
    ASSERT_NEAR(b->grad, -8.0/3.0, 1e-6);
}

TEST(ValueTest, Pow2ValuesGrad) {
    shared_ptr<Value> a = make_shared<Value>(exp(1));
    shared_ptr<Value> b = make_shared<Value>(3.0);
    shared_ptr<Value> c = pow(a, b);
	
    c->grad = 2.0;
    c->backward();

    ASSERT_NEAR(a->grad, 2*3*exp(2), 1e-6);
    ASSERT_NEAR(b->grad, 2*exp(3), 1e-6);
}

TEST(ValueTest, ManualTanh) {
    shared_ptr<Value> x = make_shared<Value>(log(2));
    shared_ptr<Value> e = make_shared<Value>(exp(1));
    shared_ptr<Value> ex = pow(e, make_shared<Value>(2.0) * x);
    shared_ptr<Value> result = (ex - make_shared<Value>(1.0)) / (ex + make_shared<Value>(1.0));

    ASSERT_NEAR(result->data, 3.0/5.0, 1e-6);
}
