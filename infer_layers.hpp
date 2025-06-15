#pragma once
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include "infer_math.hpp"

class Conv2D_Infer {
public:
    int in_c, out_c, k_size, pad, out_h, out_w;
    std::vector<float> weights, bias, output_cache, col_cache;

    Conv2D_Infer(int in_channels, int out_channels, int kernel_size, int padding, int in_h, int in_w) 
        : in_c(in_channels), out_c(out_channels), k_size(kernel_size), pad(padding) {
        out_h = in_h + 2 * pad - k_size + 1; out_w = in_w + 2 * pad - k_size + 1;
        col_cache.resize(in_c * k_size * k_size * out_h * out_w);
        output_cache.resize(out_c * out_h * out_w);
    }
    const std::vector<float>& forward(const std::vector<float>& input) {
        int patch_size = in_c * k_size * k_size, out_pixels = out_h * out_w;
        im2col_infer(input.data(), in_c, out_h - 2*pad + k_size - 1, out_w - 2*pad + k_size - 1, k_size, k_size, pad, 1, col_cache.data());
        gemm_infer(out_c, out_pixels, patch_size, weights.data(), col_cache.data(), output_cache.data());
        for (int f = 0; f < out_c; ++f) {
            float b = bias[f];
            for (int p = 0; p < out_pixels; ++p) output_cache[f * out_pixels + p] += b;
        }
        return output_cache;
    }
};

class ReLU_Infer {
public:
    std::vector<float> output_cache;
    ReLU_Infer(int size) { output_cache.resize(size); }
    const std::vector<float>& forward(const std::vector<float>& input) {
        for (size_t i = 0; i < input.size(); ++i) output_cache[i] = std::max(0.0f, input[i]);
        return output_cache;
    }
};

class MaxPool2D_Infer {
public:
    int c, h, w;
    std::vector<float> output_cache;
    MaxPool2D_Infer(int channels, int height, int width) : c(channels), h(height), w(width) {
        output_cache.resize(c * (h / 2) * (w / 2));
    }
    const std::vector<float>& forward(const std::vector<float>& input) {
        int out_h = h / 2, out_w = w / 2;
        for (int ch = 0; ch < c; ++ch) {
            for (int y = 0; y < out_h; ++y) {
                for (int x = 0; x < out_w; ++x) {
                    int out_idx = (ch * out_h + y) * out_w + x;
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (int kh = 0; kh < 2; ++kh) {
                        for (int kw = 0; kw < 2; ++kw) {
                            int in_idx = (ch * h + (y * 2 + kh)) * w + (x * 2 + kw);
                            max_val = std::max(max_val, input[in_idx]);
                        }
                    }
                    output_cache[out_idx] = max_val;
                }
            }
        }
        return output_cache;
    }
};

class Dense_Infer {
public:
    int in_f, out_f;
    std::vector<float> weights, bias, output_cache;
    Dense_Infer(int in_features, int out_features) : in_f(in_features), out_f(out_features) {
        output_cache.resize(out_f);
    }
    const std::vector<float>& forward(const std::vector<float>& input) {
        for (int i = 0; i < out_f; ++i) {
            float sum = bias[i];
            for (int j = 0; j < in_f; ++j) sum += weights[i * in_f + j] * input[j];
            output_cache[i] = sum;
        }
        return output_cache;
    }
};