#include <gtest/gtest.h>
#include "value.h"

TEST(ValueTest, CreateValue) {
	Value* a = new Value{5.0};
	ASSERT_EQ(a->data, 5.0);
}

TEST(ValueTest, Add2Values) {
	Value a{5.0};
	Value b{3.0};
	Value c = a + b;
	ASSERT_EQ(c.data, 8.0);
}

