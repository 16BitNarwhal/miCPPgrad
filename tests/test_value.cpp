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
	
	c.grad = 1.0;
	c.backward();

	ASSERT_EQ(a.grad, 1.0);
	ASSERT_EQ(b.grad, 1.0);
}

TEST(ValueTest, Mult2ValuesGrad) {
        Value a{4.0};
        Value b{7.0};
        Value c = a * b;

        c.grad = 1.0;
	c.backward();
	ASSERT_EQ(a.grad, 7.0);
	ASSERT_EQ(b.grad, 4.0);
}
