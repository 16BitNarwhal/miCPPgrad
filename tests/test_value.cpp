#include <gtest/gtest.h>
#include "value.h"

TEST(ValueTest, CreateValue) {
	Value* a = new Value{5.0};
	ASSERT_TRUE(a->data == 5.0);
}

