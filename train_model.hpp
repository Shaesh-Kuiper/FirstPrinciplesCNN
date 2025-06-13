#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include "train_layers.hpp"

class TinyVGG {
public:
    Conv2D c1, c2, c3, c4; ReLU r1, r2, r3, r4; MaxPool2D p1, p2; Dropout drop; Dense fc;
    bool is_training;

    TinyVGG() : 
        c1(1, 16, 3, 1, 28, 28), r1(16 * 28 * 28),
        c2(16, 16, 3, 1, 28, 28), r2(16 * 28 * 28),
        p1(16, 28, 28),
        c3(16, 32, 3, 1, 14, 14), r3(32 * 14 * 14),
        c4(32, 32, 3, 1, 14, 14), r4(32 * 14 * 14),
        p2(32, 14, 14),
        drop(0.3f, 32 * 7 * 7), 
        fc(32 * 7 * 7, 10), is_training(true) {}

    const std::vector<float>& forward(const std::vector<float>& x) {
        auto y = c1.forward(x);  y = r1.forward(y);
        y = c2.forward(y);       y = r2.forward(y);
        y = p1.forward(y);
        y = c3.forward(y);       y = r3.forward(y);
        y = c4.forward(y);       y = r4.forward(y);
        y = p2.forward(y);
        y = drop.forward(y, is_training); 
        return fc.forward(y);
    }

    void backward(const std::vector<float>& grad) {
        auto g = fc.backward(grad);
        g = drop.backward(g); g = p2.backward(g);
        g = r4.backward(g); g = c4.backward(g);
        g = r3.backward(g); g = c3.backward(g);
        g = p1.backward(g);
        g = r2.backward(g); g = c2.backward(g);
        g = r1.backward(g); c1.backward(g);
    }

    void sync_weights_from(const TinyVGG& o) {
        c1.sync_weights(o.c1); c2.sync_weights(o.c2);
        c3.sync_weights(o.c3); c4.sync_weights(o.c4); fc.sync_weights(o.fc);
    }
    void zero_grads() {
        c1.zero_grads(); c2.zero_grads(); c3.zero_grads(); c4.zero_grads(); fc.zero_grads();
    }
    void accumulate_grads_from(const TinyVGG& o) {
        c1.accumulate_grads(o.c1); c2.accumulate_grads(o.c2);
        c3.accumulate_grads(o.c3); c4.accumulate_grads(o.c4); fc.accumulate_grads(o.fc);
    }

    void update_weights(float lr, int batch_size) {
        c1.update_weights(lr, (float)batch_size); c2.update_weights(lr, (float)batch_size);
        c3.update_weights(lr, (float)batch_size); c4.update_weights(lr, (float)batch_size);
        fc.update_weights(lr, (float)batch_size);
    }

    void save_weights(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        auto write_layer = [&file](const std::vector<float>& w, const std::vector<float>& b) {
            int w_size = w.size(), b_size = b.size();
            file.write((char*)&w_size, sizeof(int)); file.write((char*)w.data(), w_size * sizeof(float));
            file.write((char*)&b_size, sizeof(int)); file.write((char*)b.data(), b_size * sizeof(float));
        };
        write_layer(c1.weights, c1.bias); write_layer(c2.weights, c2.bias);
        write_layer(c3.weights, c3.bias); write_layer(c4.weights, c4.bias);
        write_layer(fc.weights, fc.bias);
        std::cout << "[INFO] Model saved to " << filename << "\n";
    }
};