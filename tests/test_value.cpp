#include <gtest/gtest.h>
#include "value.h"
#include <cmath>

#include <iostream>

TEST(ValueTest, CreateValue) {
    Value a{5.0};
    ASSERT_EQ(a.data, 5.0);
}

TEST(ValueTest, Add2Values) {
    Value a{5.0};
    Value b{3.0};
    Value c = a + b;
    ASSERT_EQ(c.data, 8.0);
}

TEST(ValueTest, NegateValue) {
    Value a{5.0};
    Value b = -a;
    ASSERT_EQ(b.data, -5.0);
}

TEST(ValueTest, Sub2Values) {
    Value a{6.0};
    Value b{3.5};
    Value c = a - b;

    ASSERT_EQ(c.data, 2.5);
}

TEST(ValueTest, Mult2Values) {
    Value a{4.0};
    Value b{7.0};
    Value c = a * b;
    ASSERT_EQ(c.data, 28.0);
}

TEST(ValueTest, Div2Values) {
    Value a{12.0};
    Value b{3.0};
    Value c = a / b;
    ASSERT_EQ(c.data, 4.0);
}

TEST(ValueTest, PowValue) {
    Value a{2.0};
    Value b{3.0};
    Value c = a.pow(b);
    ASSERT_EQ(c.data, 8.0);
}

TEST(ValueTest, Add2ValuesGrad) {
    Value a{5.0};
    Value b{3.0};
    Value c = a + b;
	
    c.grad = 2.0;
    c.backward();

    ASSERT_EQ(a.grad, 2.0);
    ASSERT_EQ(b.grad, 2.0);
}

TEST(ValueTest, NegateValueGrad) {
    Value a{5.0};
    Value b = -a;
	
    b.grad = 2.0;
    b.backward();

    ASSERT_EQ(a.grad, -2.0);
}

TEST(ValueTest, Sub2ValuesGrad) {
    Value a{6.0};
    Value b{3.5};

    Value c = a - b;

    c.grad = 2.0;
    c.backward();
        
    ASSERT_EQ(a.grad, 2.0);
    ASSERT_EQ(b.grad, -2.0);
}

TEST(ValueTest, Mult2ValuesGrad) {
    Value a{4.0};
    Value b{7.0};
    Value c = a * b;

    c.grad = 2.0;
    c.backward();
    ASSERT_EQ(a.grad, 14.0);
    ASSERT_EQ(b.grad, 8.0);
}

TEST(ValueTest, Div2ValuesGrad) {
    Value a{12.0};
    Value b{3.0};
    Value c = a / b;
	
    c.grad = 2.0;
    c.backward();
    ASSERT_NEAR(a.grad, 2.0/3.0, 1e-6);
    ASSERT_NEAR(b.grad, -8.0/3.0, 1e-6);
}

TEST(ValueTest, Pow2ValuesGrad) {
    Value a{exp(1)};
    Value b{3.0};
    Value c = a.pow(b);
	
    c.grad = 2.0;
    c.backward();

    ASSERT_NEAR(a.grad, 2*3*exp(2), 1e-6);
    ASSERT_NEAR(b.grad, 2*exp(3), 1e-6);
}

TEST(ValueTest, ManualTanh) {
    Value x{log(2)};
    Value two{2.0};
    Value twox = two * x;
    Value e{exp(1)};
    Value ex = e.pow(twox);
    Value one = Value{1.0};
    Value top = ex - one;
    Value bottom = ex + one;
    Value result = top / bottom;

    ASSERT_NEAR(result.data, 3.0/5.0, 1e-6);

    result.grad = 1;
    result.backward();
    top.backward();
    bottom.backward();
    ex.backward();
    twox.backward();

    ASSERT_NEAR(x.data, 16.0/25.0, 1e-1);
}
