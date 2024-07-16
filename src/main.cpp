#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <chrono>
#include "neural_network.h"
using namespace std;

const int EPOCHS = 10000;
const int NUM_SAVES = 100;
const double LEARNING_RATE = 0.03;
const vector<pair<int, Activation>> LAYERS_CONFIG = {
    {2, NONE},
    {32, RELU},
    {8, RELU},
    {4, RELU},
    {1, NONE}
};

shared_ptr<Value> mse(shared_ptr<Value>& ypred, shared_ptr<Value>& y) {
    shared_ptr<Value> diff = ypred - y;
    return pow(diff, make_shared<Value>(2.0));
}

double ground_truth(double x, double y) {
    return 2.0*sin(x*3.14) - y + 0.5;
}

int main() {
    // sample dataset
    vector<vector<double>> X;
    for (double x=-1; x<=1; x+=0.3) {
        for (double y=-1; y<=1; y+=0.3) {
            X.push_back({x,y});
        }
    }
    vector<double> y;
    for (vector<double> x : X) {
        y.push_back(ground_truth(x[0], x[1]));
    }
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
    MultiLayerPerceptron model(LAYERS_CONFIG);

    // file for writing
    ofstream outFile("out/neural_network_3d_graph.csv");
    outFile << "Epoch,Loss,X,Y,Z_pred,Z_true" << endl;

    auto totalDuration = 0.0;
    // train loop
    double lr = LEARNING_RATE;
    for (int i = 1; i <= EPOCHS; i++) {
        auto start = chrono::steady_clock::now();

        // gradient descent
        model.zero_grad();
        vector<shared_ptr<Value>> ypred;
        for (auto& x_ : Xval) {
            ypred.push_back(model(x_)[0]);
        }
        shared_ptr<Value> loss = make_shared<Value>(0.0);
        for (int j = 0; j < ypred.size(); j++) {
            loss = loss + mse(ypred[j], yval[j]);
        }
        loss = loss / make_shared<Value>(ypred.size());
        loss->backprop();
        for (auto& p : model.get_parameters()) {
            p->data -= p->grad * lr;
        }

        auto end = chrono::steady_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        totalDuration += duration;

        // saving predictions
        if (i % (EPOCHS / NUM_SAVES) == 0) {
            cout << "epoch: " << i << ", loss: " << loss->data << ", total duration: " << totalDuration << "ms" << endl;

            for (double x = -1; x <= 1; x += 0.1) {
                for (double y = -1; y <= 1; y += 0.1) {
                    vector<shared_ptr<Value>> input = {make_shared<Value>(x), make_shared<Value>(y)};
                    shared_ptr<Value> z_pred = model(input)[0];
                    double z_true = ground_truth(x, y);
                    outFile << i << "," << loss->data << "," << x << "," << y << "," << z_pred->data << "," << z_true << endl;
                }
            }
        }
    }

    outFile.close();

    // evaluate
    vector<shared_ptr<Value>> xval = {make_shared<Value>(0.4), make_shared<Value>(0.3)};
    shared_ptr<Value> ypred = model(xval)[0];
    std::cout << "ypred: " << ypred->data << endl;
    std::cout << "ytrue: " << ground_truth(0.4, 0.3) << endl;
}