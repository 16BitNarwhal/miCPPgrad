#include <gtest/gtest.h>
#include "value.h"

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

TEST(ValueTest, Mult2Values) {
	Value a{4.0};
	Value b{7.0};
	Value c = a * b;
	ASSERT_EQ(c.data, 28.0);
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

TEST(ValueTest, NegativeValueGrad) {
	Value a{5.0};
	Value b = -a;
	
	b.grad = 2.0;
	b.backward();

	ASSERT_EQ(a.grad, -2.0);
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
