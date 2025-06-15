#pragma once
#include <algorithm>

inline void im2col_infer(const float* im, int channels, int height, int width, int kh, int kw, int pad, int stride, float* col) {
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

inline void gemm_infer(int M, int N, int K, const float* A, const float* B, float* C) {
    std::fill(C, C + M * N, 0.0f);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}