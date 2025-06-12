#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>
#include "train_math.hpp"

class Conv2D {
public:
    int in_c, out_c, k_size, pad, in_h, in_w, out_h, out_w, t;
    std::vector<float> weights, bias, grad_W, grad_b, m_W, v_W, m_b, v_b;
    std::vector<float> output_cache, col_cache, grad_col_cache, grad_in_cache;

    Conv2D(int in_channels, int out_channels, int kernel_size, int padding, int h, int w) 
        : in_c(in_channels), out_c(out_channels), k_size(kernel_size), pad(padding), in_h(h), in_w(w), t(0) {
        out_h = in_h + 2 * pad - k_size + 1;
        out_w = in_w + 2 * pad - k_size + 1;
        
        int w_size = out_c * in_c * k_size * k_size;
        weights.resize(w_size); bias.assign(out_c, 0.0f);
        grad_W.assign(w_size, 0.0f); grad_b.assign(out_c, 0.0f);
        m_W.assign(w_size, 0.0f); v_W.assign(w_size, 0.0f);
        m_b.assign(out_c, 0.0f); v_b.assign(out_c, 0.0f);

        int patch_size = in_c * k_size * k_size;
        int out_pixels = out_h * out_w;
        col_cache.resize(patch_size * out_pixels);
        output_cache.resize(out_c * out_pixels);
        grad_col_cache.resize(patch_size * out_pixels);
        grad_in_cache.resize(in_c * in_h * in_w);

        float stddev = std::sqrt(2.0f / patch_size);
        std::mt19937 gen(42 + in_channels + out_channels);
        std::normal_distribution<float> d(0.0f, stddev);
        for (int i = 0; i < w_size; ++i) weights[i] = d(gen);
    }

    const std::vector<float>& forward(const std::vector<float>& input) {
        int patch_size = in_c * k_size * k_size, out_pixels = out_h * out_w;
        im2col(input.data(), in_c, in_h, in_w, k_size, k_size, pad, 1, col_cache.data());
        gemm(out_c, out_pixels, patch_size, weights.data(), col_cache.data(), output_cache.data());
        for (int f = 0; f < out_c; ++f) {
            float b = bias[f];
            for (int p = 0; p < out_pixels; ++p) output_cache[f * out_pixels + p] += b;
        }
        return output_cache;
    }

    const std::vector<float>& backward(const std::vector<float>& grad_out) {
        int out_pixels = out_h * out_w, patch_size = in_c * k_size * k_size;
        for (int f = 0; f < out_c; ++f) {
            float sum = 0.0f;
            for (int p = 0; p < out_pixels; ++p) sum += grad_out[f * out_pixels + p];
            grad_b[f] += sum;
        }
        gemm_A_BT_accumulate(out_c, patch_size, out_pixels, grad_out.data(), col_cache.data(), grad_W.data());
        gemm_AT_B(patch_size, out_pixels, out_c, weights.data(), grad_out.data(), grad_col_cache.data());
        col2im(grad_col_cache.data(), in_c, in_h, in_w, k_size, k_size, pad, 1, grad_in_cache.data());
        return grad_in_cache;
    }

    void sync_weights(const Conv2D& o) { weights = o.weights; bias = o.bias; }
    void zero_grads() { std::fill(grad_W.begin(), grad_W.end(), 0.0f); std::fill(grad_b.begin(), grad_b.end(), 0.0f); }
    void accumulate_grads(const Conv2D& o) {
        for(size_t i=0; i<grad_W.size(); ++i) grad_W[i] += o.grad_W[i];
        for(size_t i=0; i<grad_b.size(); ++i) grad_b[i] += o.grad_b[i];
    }

    void update_weights(float lr, float batch_size) {
        t++;
        const float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, weight_decay = 1e-4f;
        float bias_corr1 = 1.0f - std::pow(beta1, t), bias_corr2 = 1.0f - std::pow(beta2, t);
        for (size_t i = 0; i < weights.size(); ++i) {
            float g = (grad_W[i] / batch_size) + weight_decay * weights[i];
            m_W[i] = beta1 * m_W[i] + (1.0f - beta1) * g; v_W[i] = beta2 * v_W[i] + (1.0f - beta2) * g * g;
            weights[i] -= lr * (m_W[i] / bias_corr1) / (std::sqrt(v_W[i] / bias_corr2) + eps);
        }
        for (size_t i = 0; i < bias.size(); ++i) {
            float g = grad_b[i] / batch_size;
            m_b[i] = beta1 * m_b[i] + (1.0f - beta1) * g; v_b[i] = beta2 * v_b[i] + (1.0f - beta2) * g * g;
            bias[i] -= lr * (m_b[i] / bias_corr1) / (std::sqrt(v_b[i] / bias_corr2) + eps);
        }
    }
};

class ReLU {
public:
    std::vector<float> input_cache, output_cache, grad_in_cache;
    ReLU(int size) { input_cache.resize(size); output_cache.resize(size); grad_in_cache.resize(size); }
    const std::vector<float>& forward(const std::vector<float>& input) {
        std::copy(input.begin(), input.end(), input_cache.begin());
        for (size_t i = 0; i < input.size(); ++i) output_cache[i] = std::max(0.0f, input[i]);
        return output_cache;
    }
    const std::vector<float>& backward(const std::vector<float>& grad_out) {
        for (size_t i = 0; i < grad_out.size(); ++i) grad_in_cache[i] = input_cache[i] > 0.0f ? grad_out[i] : 0.0f;
        return grad_in_cache;
    }
};

class MaxPool2D {
public:
    int c, h, w;
    std::vector<int> max_indices;
    std::vector<float> output_cache, grad_in_cache;
    MaxPool2D(int channels, int height, int width) : c(channels), h(height), w(width) {
        int out_h = h / 2, out_w = w / 2;
        max_indices.resize(c * out_h * out_w);
        output_cache.resize(c * out_h * out_w);
        grad_in_cache.resize(c * h * w);
    }
    const std::vector<float>& forward(const std::vector<float>& input) {
        int out_h = h / 2, out_w = w / 2;
        for (int ch = 0; ch < c; ++ch) {
            for (int y = 0; y < out_h; ++y) {
                for (int x = 0; x < out_w; ++x) {
                    int out_idx = (ch * out_h + y) * out_w + x;
                    float max_val = -std::numeric_limits<float>::infinity();
                    int max_idx = -1;
                    for (int kh = 0; kh < 2; ++kh) {
                        for (int kw = 0; kw < 2; ++kw) {
                            int in_idx = (ch * h + (y * 2 + kh)) * w + (x * 2 + kw);
                            if (input[in_idx] > max_val) { max_val = input[in_idx]; max_idx = in_idx; }
                        }
                    }
                    output_cache[out_idx] = max_val;
                    max_indices[out_idx] = max_idx;
                }
            }
        }
        return output_cache;
    }
    const std::vector<float>& backward(const std::vector<float>& grad_out) {
        std::fill(grad_in_cache.begin(), grad_in_cache.end(), 0.0f);
        for (size_t i = 0; i < grad_out.size(); ++i) grad_in_cache[max_indices[i]] = grad_out[i];
        return grad_in_cache;
    }
};

class Dropout {
public:
    float rate;
    std::vector<float> mask, output_cache, grad_in_cache;
    Dropout(float drop_rate, int size) : rate(drop_rate) {
        mask.resize(size); output_cache.resize(size); grad_in_cache.resize(size);
    }
    const std::vector<float>& forward(const std::vector<float>& input, bool is_training) {
        if (!is_training) return input;
        float scale = 1.0f / (1.0f - rate);
        thread_local std::mt19937 gen(std::random_device{}());
        std::bernoulli_distribution d(1.0f - rate); 
        for (size_t i = 0; i < input.size(); ++i) {
            mask[i] = d(gen) ? 1.0f : 0.0f;
            output_cache[i] = input[i] * mask[i] * scale;
        }
        return output_cache;
    }
    const std::vector<float>& backward(const std::vector<float>& grad_out) {
        float scale = 1.0f / (1.0f - rate);
        for (size_t i = 0; i < grad_out.size(); ++i) grad_in_cache[i] = grad_out[i] * mask[i] * scale;
        return grad_in_cache;
    }
};

class Dense {
public:
    int in_f, out_f, t;
    std::vector<float> weights, bias, grad_W, grad_b, m_W, v_W, m_b, v_b;
    std::vector<float> input_cache, output_cache, grad_in_cache;

    Dense(int in_features, int out_features) : in_f(in_features), out_f(out_features), t(0) {
        int w_size = out_f * in_f;
        weights.resize(w_size); bias.assign(out_f, 0.0f);
        grad_W.assign(w_size, 0.0f); grad_b.assign(out_f, 0.0f);
        m_W.assign(w_size, 0.0f); v_W.assign(w_size, 0.0f);
        m_b.assign(out_f, 0.0f); v_b.assign(out_f, 0.0f);
        
        input_cache.resize(in_f); output_cache.resize(out_f); grad_in_cache.resize(in_f);

        float stddev = std::sqrt(2.0f / in_f);
        std::mt19937 gen(99);
        std::normal_distribution<float> d(0.0f, stddev);
        for (int i = 0; i < w_size; ++i) weights[i] = d(gen);
    }
    const std::vector<float>& forward(const std::vector<float>& input) {
        std::copy(input.begin(), input.end(), input_cache.begin());
        std::fill(output_cache.begin(), output_cache.end(), 0.0f);
        for (int i = 0; i < out_f; ++i) {
            float sum = bias[i];
            for (int j = 0; j < in_f; ++j) sum += weights[i * in_f + j] * input[j];
            output_cache[i] = sum;
        }
        return output_cache;
    }
    const std::vector<float>& backward(const std::vector<float>& grad_out) {
        std::fill(grad_in_cache.begin(), grad_in_cache.end(), 0.0f);
        for (int i = 0; i < out_f; ++i) {
            float g = grad_out[i];
            grad_b[i] += g;
            for (int j = 0; j < in_f; ++j) {
                grad_W[i * in_f + j] += g * input_cache[j];
                grad_in_cache[j] += g * weights[i * in_f + j];
            }
        }
        return grad_in_cache;
    }

    void sync_weights(const Dense& o) { weights = o.weights; bias = o.bias; }
    void zero_grads() { std::fill(grad_W.begin(), grad_W.end(), 0.0f); std::fill(grad_b.begin(), grad_b.end(), 0.0f); }
    void accumulate_grads(const Dense& o) {
        for(size_t i=0; i<grad_W.size(); ++i) grad_W[i] += o.grad_W[i];
        for(size_t i=0; i<grad_b.size(); ++i) grad_b[i] += o.grad_b[i];
    }

    void update_weights(float lr, float batch_size) {
        t++;
        const float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, weight_decay = 1e-4f;
        float bias_corr1 = 1.0f - std::pow(beta1, t), bias_corr2 = 1.0f - std::pow(beta2, t);
        for (size_t i = 0; i < weights.size(); ++i) {
            float g = (grad_W[i] / batch_size) + weight_decay * weights[i];
            m_W[i] = beta1 * m_W[i] + (1.0f - beta1) * g; v_W[i] = beta2 * v_W[i] + (1.0f - beta2) * g * g;
            weights[i] -= lr * (m_W[i] / bias_corr1) / (std::sqrt(v_W[i] / bias_corr2) + eps);
        }
        for (size_t i = 0; i < bias.size(); ++i) {
            float g = grad_b[i] / batch_size;
            m_b[i] = beta1 * m_b[i] + (1.0f - beta1) * g; v_b[i] = beta2 * v_b[i] + (1.0f - beta2) * g * g;
            bias[i] -= lr * (m_b[i] / bias_corr1) / (std::sqrt(v_b[i] / bias_corr2) + eps);
        }
    }
};