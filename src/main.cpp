#include <iostream>
#include "neural_network.h"

using namespace std;

shared_ptr<Value> mse(shared_ptr<Value>& ypred, shared_ptr<Value>& y) {
    return (ypred - y) * (ypred - y);
}

int main() {
    // sample dataset
    vector<vector<double>> X = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    vector<double> y = {0.0, 1.0, 1.0, 2.0};

    // convert to Value
    vector<vector<shared_ptr<Value>>> Xval;
    for (auto &x_ : X) {
        Xval.push_back(vector<shared_ptr<Value>> {make_shared<Value>(x_[0]), make_shared<Value>(x_[1])});
    }
    vector<shared_ptr<Value>> yval;
    for (auto &y_ : y) {
        yval.push_back(make_shared<Value>(y_));
    }

    // create model
    MultiLayerPerceptron model({2, 3, 1});

    // train loop
    double lr = 0.03;
    int epochs = 5000;
    for (int i = 1; i <= epochs; i++) {
        // gradient descent
        model.zero_grad();
        vector<shared_ptr<Value>> ypred;
        for (auto& x_ : Xval) {
            ypred.push_back(model(x_)[0]);
        }
        shared_ptr<Value> loss = make_shared<Value>(0.0);
        for (int i = 0; i < ypred.size(); i++) {
            loss = loss + mse(ypred[i], yval[i]);
        }
        loss->backprop();
        for (auto& p : model.get_parameters()) {
            p->data -= p->grad * lr;
        }
        if (i % (epochs / 10) == 0) {
            std::cout << "loss on epoch " << i << ": " << loss->data << endl;
        }
    }

    // evaluate
    vector<shared_ptr<Value>> xval = {make_shared<Value>(2.0), make_shared<Value>(2.0)};
    shared_ptr<Value> ypred = model(xval)[0];
    std::cout << "ypred: " << ypred->data << endl;
}
