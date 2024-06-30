#include <iostream>
#include "value.h"

int main() {

	Value a{2.0};
	Value b{3.0};
	Value c = a + b;
	std::cout << "c.data: " << c.data << "\n";

}

