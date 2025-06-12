#pragma once
#include <cstdint>
#include <algorithm>

inline uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0x00FF00FF);
    return (val << 16) | (val >> 16);
}

inline void gemm(int M, int N, int K, const float* A, const float* B, float* C) {
    std::fill(C, C + M * N, 0.0f);
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float a = A[i * K + k];
            for (int j = 0; j < N; ++j) C[i * N + j] += a * B[k * N + j];
        }
    }
}

inline void gemm_A_BT_accumulate(int M, int N, int K, const float* A, const float* B, float* C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) sum += A[i * K + k] * B[j * K + k];
            C[i * N + j] += sum;
        }
    }
}

inline void gemm_AT_B(int M, int N, int K, const float* A, const float* B, float* C) {
    std::fill(C, C + M * N, 0.0f);
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < M; ++i) {
            float a = A[k * M + i];
            for (int j = 0; j < N; ++j) C[i * N + j] += a * B[k * N + j];
        }
    }
}

inline void im2col(const float* im, int channels, int height, int width, int kh, int kw, int pad, int stride, float* col) {
    int out_h = (height + 2 * pad - kh) / stride + 1;
    int out_w = (width + 2 * pad - kw) / stride + 1;
    int channels_col = channels * kh * kw;
    for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % kw, h_offset = (c / kw) % kh, c_im = c / kh / kw;
        for (int h = 0; h < out_h; ++h) {
            for (int w = 0; w < out_w; ++w) {
                int im_row = h_offset + h * stride - pad;
                int im_col = w_offset + w * stride - pad;
                int col_idx = c * (out_h * out_w) + h * out_w + w;
                col[col_idx] = (im_row >= 0 && im_col >= 0 && im_row < height && im_col < width) ? 
                               im[(c_im * height + im_row) * width + im_col] : 0.0f;
            }
        }
    }
}

inline void col2im(const float* col, int channels, int height, int width, int kh, int kw, int pad, int stride, float* im) {
    std::fill(im, im + channels * height * width, 0.0f);
    int out_h = (height + 2 * pad - kh) / stride + 1;
    int out_w = (width + 2 * pad - kw) / stride + 1;
    int channels_col = channels * kh * kw;
    for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % kw, h_offset = (c / kw) % kh, c_im = c / kh / kw;
        for (int h = 0; h < out_h; ++h) {
            for (int w = 0; w < out_w; ++w) {
                int im_row = h_offset + h * stride - pad;
                int im_col = w_offset + w * stride - pad;
                int col_idx = c * (out_h * out_w) + h * out_w + w;
                if (im_row >= 0 && im_col >= 0 && im_row < height && im_col < width) {
                    im[(c_im * height + im_row) * width + im_col] += col[col_idx];
                }
            }
        }
    }
}