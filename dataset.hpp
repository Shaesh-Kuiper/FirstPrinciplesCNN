#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>
#include "train_math.hpp" // Needs swap_endian

inline void random_horizontal_flip(std::vector<float>& img, int h = 28, int w = 28) {
    thread_local std::mt19937 gen(std::random_device{}());
    thread_local std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    if (dist(gen) > 0.5f) {
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w / 2; ++j) std::swap(img[i * w + j], img[i * w + (w - 1 - j)]);
        }
    }
}

inline std::vector<std::vector<float>> read_mnist_images(const std::string& path, int& count) {
    std::ifstream file(path, std::ios::binary);
    if(!file) { std::cerr << "Missing " << path << "\n"; exit(1); }
    uint32_t magic = 0, num_items = 0, rows = 0, cols = 0;
    file.read((char*)&magic, 4); file.read((char*)&num_items, 4);
    file.read((char*)&rows, 4);  file.read((char*)&cols, 4);
    num_items = swap_endian(num_items); rows = swap_endian(rows); cols = swap_endian(cols);
    count = num_items;
    
    std::vector<std::vector<float>> images(num_items, std::vector<float>(rows * cols));
    std::vector<uint8_t> buffer(rows * cols);
    const float mean = 0.286f, std = 0.353f;

    for (uint32_t i = 0; i < num_items; ++i) {
        file.read((char*)buffer.data(), rows * cols);
        for (uint32_t j = 0; j < rows * cols; ++j) images[i][j] = ((buffer[j] / 255.0f) - mean) / std; 
    }
    return images;
}

inline std::vector<int> read_mnist_labels(const std::string& path, int& count) {
    std::ifstream file(path, std::ios::binary);
    uint32_t magic = 0, num_items = 0;
    file.read((char*)&magic, 4); file.read((char*)&num_items, 4);
    num_items = swap_endian(num_items); count = num_items;
    std::vector<int> labels(num_items);
    std::vector<uint8_t> buffer(num_items);
    file.read((char*)buffer.data(), num_items);
    for (uint32_t i = 0; i < num_items; ++i) labels[i] = buffer[i];
    return labels;
}

inline float cross_entropy(const std::vector<float>& logits, int label, std::vector<float>& grad) {
    float max_logit = *std::max_element(logits.begin(), logits.end()), sum = 0.0f;
    std::vector<float> exps(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) { exps[i] = std::exp(logits[i] - max_logit); sum += exps[i]; }
    float loss = -std::log(std::max(exps[label] / sum, 1e-15f));
    grad.resize(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) grad[i] = (exps[i] / sum);
    grad[label] -= 1.0f;
    return loss;
}