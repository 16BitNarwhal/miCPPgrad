#include <iostream>
#include <memory>
#include <assert.h>
#include "value.h"

using namespace std;

int main() {
    shared_ptr<Value> a = make_shared<Value>(6.0);
    shared_ptr<Value> b = make_shared<Value>(3.5);

    shared_ptr<Value> c = a + b;
}
