#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include "infer_model.hpp"

std::vector<float> load_and_preprocess_image(const std::string& path) {
    int w, h, channels;
    unsigned char* img = stbi_load(path.c_str(), &w, &h, &channels, 1); 
    if (!img) {
        std::cerr << "Error loading image: " << path << "\n";
        return {};
    }

    unsigned char resized_img[28 * 28];
    stbir_resize_uint8(img, w, h, 0, resized_img, 28, 28, 0, 1);
    stbi_image_free(img);

    int corner_sum = resized_img[0] + resized_img[27] + 
                     resized_img[27 * 28] + resized_img[27 * 28 + 27];
    bool invert = (corner_sum / 4) > 127; 

    if (invert) {
        std::cout << "[INFO] Light background detected. Inverting colors to match Fashion MNIST...\n";
    }

    std::cout << "\n--- WHAT THE NEURAL NETWORK SEES ---\n";
    std::vector<float> out(28 * 28);
    const float mean = 0.286f, std_dev = 0.353f;

    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            int idx = y * 28 + x;
            float pixel_val = resized_img[idx];
            
            if (invert) pixel_val = 255.0f - pixel_val;

            if (pixel_val > 180) std::cout << "@@";
            else if (pixel_val > 80) std::cout << "::";
            else std::cout << "  ";

            out[idx] = ((pixel_val / 255.0f) - mean) / std_dev; 
        }
        std::cout << "\n";
    }
    std::cout << "------------------------------------\n";

    return out;
}

int main() {
    TinyVGG_Infer model;
    std::cout << "Loading model from tinyvgg_fashion_mnist.bin...\n";
    if (!model.load_weights("tinyvgg_fashion_mnist.bin")) {
        std::cerr << "Failed to load model. Did you run the training script first?\n";
        return 1;
    }
    std::cout << "Model loaded successfully.\n\n";

    const char* classes[] = {"T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
                             "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"};

    std::string filepath;
    while (true) {
        std::cout << "Enter image file path (or 'exit' to quit): ";
        std::getline(std::cin, filepath);
        if (filepath == "exit" || filepath == "quit") break;

        std::vector<float> img_tensor = load_and_preprocess_image(filepath);
        if (img_tensor.empty()) continue;

        auto logits = model.forward(img_tensor);
        int pred = std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));

        std::cout << "--> Prediction: " << classes[pred] << " (Class ID " << pred << ")\n\n";
    }

    return 0;
}