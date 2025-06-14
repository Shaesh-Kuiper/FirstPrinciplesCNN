#include <iostream>
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

    float accuracy = 0;
    for (int i = 0; i < test_count; ++i) {
        int id = indices[train_count + i];
        auto logits = thread_models[0].forward(all_images[id]); 
        int pred = std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
        if(pred == all_labels[id]) accuracy++;
    }

    std::cout << "Final Test Accuracy: " << (accuracy / test_count) * 100.0f << "%\n";
    thread_models[0].save_weights("tinyvgg_fashion_mnist.bin");

    return 0;
}