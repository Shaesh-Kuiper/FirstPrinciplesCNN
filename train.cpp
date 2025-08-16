#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <omp.h>
#include "train_model.hpp"
#include "dataset.hpp"

struct Shape { int c, h, w; };

int main() {
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    std::cout << "[INFO] Data Parallelism enabled across " << num_threads << " CPU threads.\n";

    int total_img = 0, total_lbl = 0;
    auto all_images = read_mnist_images("train-images-idx3-ubyte", total_img);
    auto all_labels = read_mnist_labels("train-labels-idx1-ubyte", total_lbl);

    std::vector<int> indices(total_img);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 g(42); 

    int train_count = static_cast<int>(total_img * 0.85);
    int test_count = total_img - train_count;
    std::cout << "[INFO] Dataset loaded. Training: " << train_count << " | Testing: " << test_count << "\n";

    std::vector<TinyVGG> thread_models(num_threads);
    for (int t = 1; t < num_threads; ++t) thread_models[t].sync_weights_from(thread_models[0]);

    int epochs = 50; 
    int batch_size = 64; 
    float lr = 0.0028f; 

    for (int ep = 1; ep <= epochs; ++ep) {
        float epoch_loss = 0.0f;
        std::shuffle(indices.begin(), indices.begin() + train_count, g);

        if (ep > 1 && ep % 6 == 0) {
            lr *= 0.5f;
            std::cout << "[INFO] Learning Rate decayed to: " << lr << "\n";
        }

        for (int b = 0; b < train_count; b += batch_size) {
            int current_batch_size = std::min(batch_size, train_count - b);
            
            for (int t = 0; t < num_threads; ++t) thread_models[t].zero_grads();

            #pragma omp parallel for reduction(+:epoch_loss)
            for (int i = 0; i < current_batch_size; ++i) {
                int tid = omp_get_thread_num();
                int id = indices[b + i];
                
                std::vector<float> augmented_img = all_images[id];
                random_horizontal_flip(augmented_img);

                auto logits = thread_models[tid].forward(augmented_img);
                std::vector<float> loss_grad;
                epoch_loss += cross_entropy(logits, all_labels[id], loss_grad);
                
                thread_models[tid].backward(loss_grad);
            }

            for (int t = 1; t < num_threads; ++t) {
                thread_models[0].accumulate_grads_from(thread_models[t]);
            }

            thread_models[0].update_weights(lr, current_batch_size);

            for (int t = 1; t < num_threads; ++t) {
                thread_models[t].sync_weights_from(thread_models[0]);
            }
        }
        std::cout << "Epoch " << ep << "/" << epochs << " - Average Loss: " << (epoch_loss / train_count) << "\n";
    }

    std::cout << "[INFO] Validating against holdout set...\n";
    thread_models[0].is_training = false;

    const int NUM_CLASSES = 10;
    const char* CLASS_NAMES[NUM_CLASSES] = {
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    };
    const char* SHORT_NAMES[NUM_CLASSES] = {
        "T-sht", "Trsr", "Pull", "Drss", "Coat",
        "Sndl", "Shrt", "Snkr", "Bag", "Ankl"
    };

    std::vector<std::vector<int>> conf(NUM_CLASSES, std::vector<int>(NUM_CLASSES, 0));
    int correct = 0;

    for (int i = 0; i < test_count; ++i) {
        int id = indices[train_count + i];
        auto logits = thread_models[0].forward(all_images[id]);
        int pred = std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
        conf[all_labels[id]][pred]++;
        if (pred == all_labels[id]) correct++;
    }

    std::vector<int> row_sum(NUM_CLASSES, 0), col_sum(NUM_CLASSES, 0);
    for (int i = 0; i < NUM_CLASSES; ++i)
        for (int j = 0; j < NUM_CLASSES; ++j) {
            row_sum[i] += conf[i][j];
            col_sum[j] += conf[i][j];
        }

    std::vector<float> prec(NUM_CLASSES), rec(NUM_CLASSES), f1(NUM_CLASSES);
    for (int c = 0; c < NUM_CLASSES; ++c) {
        prec[c] = (col_sum[c] > 0) ? (float)conf[c][c] / col_sum[c] : 0.0f;
        rec[c]  = (row_sum[c] > 0) ? (float)conf[c][c] / row_sum[c] : 0.0f;
        f1[c]   = (prec[c] + rec[c] > 0.0f) ? 2.0f * prec[c] * rec[c] / (prec[c] + rec[c]) : 0.0f;
    }

    float macro_p = 0.0f, macro_r = 0.0f, macro_f1 = 0.0f;
    for (int c = 0; c < NUM_CLASSES; ++c) { macro_p += prec[c]; macro_r += rec[c]; macro_f1 += f1[c]; }
    macro_p /= NUM_CLASSES; macro_r /= NUM_CLASSES; macro_f1 /= NUM_CLASSES;

    const int LBL_W = 13, CELL_W = 6;
    const std::string SEP(LBL_W + 1 + NUM_CLASSES * CELL_W + 2, '-');

    std::cout << "\n=== CONFUSION MATRIX (rows=Actual, cols=Predicted) ===\n\n";
    std::cout << std::setw(LBL_W + 1) << "";
    for (int j = 0; j < NUM_CLASSES; ++j) std::cout << std::setw(CELL_W) << SHORT_NAMES[j];
    std::cout << "\n" << SEP << "\n";
    for (int i = 0; i < NUM_CLASSES; ++i) {
        std::cout << std::right << std::setw(LBL_W) << CLASS_NAMES[i] << "|";
        for (int j = 0; j < NUM_CLASSES; ++j) std::cout << std::setw(CELL_W) << conf[i][j];
        std::cout << " |\n";
    }
    std::cout << SEP << "\n";

    std::cout << "\n=== PER-CLASS METRICS ===\n\n";
    std::cout << std::left  << std::setw(14) << "Class"
              << std::right << std::setw(11) << "Precision"
              << std::setw(9)  << "Recall"
              << std::setw(11) << "F1-Score"
              << std::setw(9)  << "Support" << "\n";
    std::cout << std::string(54, '-') << "\n";
    std::cout << std::fixed << std::setprecision(4);
    for (int c = 0; c < NUM_CLASSES; ++c) {
        std::cout << std::left  << std::setw(14) << CLASS_NAMES[c]
                  << std::right << std::setw(11) << prec[c]
                  << std::setw(9)  << rec[c]
                  << std::setw(11) << f1[c]
                  << std::setw(9)  << row_sum[c] << "\n";
    }
    std::cout << std::string(54, '-') << "\n";
    std::cout << std::left  << std::setw(14) << "Macro Avg"
              << std::right << std::setw(11) << macro_p
              << std::setw(9)  << macro_r
              << std::setw(11) << macro_f1
              << std::setw(9)  << test_count << "\n";

    std::cout << "\nFinal Test Accuracy: " << std::setprecision(2)
              << (float)correct / test_count * 100.0f << "%\n";
    thread_models[0].save_weights("tinyvgg_fashion_mnist.bin");

    return 0;
}