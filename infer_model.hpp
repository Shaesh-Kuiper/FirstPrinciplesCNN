#pragma once
#include <vector>
#include <string>
#include <fstream>
#include "infer_layers.hpp"

class TinyVGG_Infer {
public:
    Conv2D_Infer c1, c2, c3, c4;
    ReLU_Infer r1, r2, r3, r4;
    MaxPool2D_Infer p1, p2;
    Dense_Infer fc;

    TinyVGG_Infer() : 
        c1(1, 16, 3, 1, 28, 28), r1(16 * 28 * 28),
        c2(16, 16, 3, 1, 28, 28), r2(16 * 28 * 28),
        p1(16, 28, 28),
        c3(16, 32, 3, 1, 14, 14), r3(32 * 14 * 14),
        c4(32, 32, 3, 1, 14, 14), r4(32 * 14 * 14),
        p2(32, 14, 14), fc(32 * 7 * 7, 10) {}

    const std::vector<float>& forward(const std::vector<float>& x) {
        auto y = c1.forward(x);  y = r1.forward(y);
        y = c2.forward(y);       y = r2.forward(y);
        y = p1.forward(y);
        y = c3.forward(y);       y = r3.forward(y);
        y = c4.forward(y);       y = r4.forward(y);
        y = p2.forward(y);
        return fc.forward(y);
    }

    bool load_weights(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if(!file) return false;
        
        auto read_layer = [&file](std::vector<float>& w, std::vector<float>& b) {
            int w_size, b_size;
            file.read((char*)&w_size, sizeof(int));
            w.resize(w_size); file.read((char*)w.data(), w_size * sizeof(float));
            file.read((char*)&b_size, sizeof(int));
            b.resize(b_size); file.read((char*)b.data(), b_size * sizeof(float));
        };

        read_layer(c1.weights, c1.bias); read_layer(c2.weights, c2.bias);
        read_layer(c3.weights, c3.bias); read_layer(c4.weights, c4.bias);
        read_layer(fc.weights, fc.bias);
        return true;
    }
};