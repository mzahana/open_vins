/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <bitset>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <input_ov_secondary.bin> <output_openvins.bin>" << std::endl;
        return -1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];

    // Read ov_secondary format
    std::ifstream input(input_file, std::ios::binary);
    if (!input.is_open()) {
        std::cerr << "Cannot open input file: " << input_file << std::endl;
        return -1;
    }

    try {
        // Read ov_secondary header (first 8 bytes are k=10, L=6)
        int k, L;
        input.read(reinterpret_cast<char*>(&k), sizeof(k));
        input.read(reinterpret_cast<char*>(&L), sizeof(L));

        std::cout << "ov_secondary vocabulary: k=" << k << ", L=" << L << std::endl;

        // Calculate total nodes (assuming DBoW2 structure)
        // For k=10, L=6: approximately 10^6 = 1M leaf nodes
        int total_words = 1;
        for (int i = 0; i < L; ++i) {
            total_words *= k;
        }
        std::cout << "Expected vocabulary size: " << total_words << " words" << std::endl;

        // Skip rest of header and read descriptors directly
        // ov_secondary format: [k][L][padding][descriptor1][descriptor2]...
        input.seekg(16, std::ios::beg); // Skip to descriptors

        std::vector<std::bitset<256>> descriptors;
        char buffer[32]; // 256 bits = 32 bytes

        int count = 0;
        while (input.read(buffer, 32) && count < total_words) {
            std::bitset<256> desc;
            for (int i = 0; i < 32; ++i) {
                for (int j = 0; j < 8; ++j) {
                    if (buffer[i] & (1 << j)) {
                        desc[i * 8 + j] = 1;
                    }
                }
            }
            descriptors.push_back(desc);
            count++;
        }

        std::cout << "Read " << descriptors.size() << " descriptors" << std::endl;
        input.close();

        // Write OpenVINS format
        std::ofstream output(output_file, std::ios::binary);
        if (!output.is_open()) {
            std::cerr << "Cannot open output file: " << output_file << std::endl;
            return -1;
        }

        // Write OpenVINS format:
        // [num_nodes][node1][node2]...
        // node = [id][descriptor_32bytes][weight][num_children][children_ids...][is_leaf]

        int num_nodes = descriptors.size();
        output.write(reinterpret_cast<const char*>(&num_nodes), sizeof(num_nodes));

        for (int i = 0; i < num_nodes; ++i) {
            // Node ID
            output.write(reinterpret_cast<const char*>(&i), sizeof(i));

            // Descriptor (32 bytes)
            std::string desc_str(32, '\0');
            for (int j = 0; j < 32; ++j) {
                char byte = 0;
                for (int bit = 0; bit < 8; ++bit) {
                    if (descriptors[i][j * 8 + bit]) {
                        byte |= (1 << bit);
                    }
                }
                desc_str[j] = byte;
            }
            output.write(desc_str.c_str(), 32);

            // Weight (default to 1.0 for leaf nodes)
            double weight = 1.0;
            output.write(reinterpret_cast<const char*>(&weight), sizeof(weight));

            // Number of children (0 for leaf nodes)
            int num_children = 0;
            output.write(reinterpret_cast<const char*>(&num_children), sizeof(num_children));

            // is_leaf flag (true for all words)
            bool is_leaf = true;
            output.write(reinterpret_cast<const char*>(&is_leaf), sizeof(is_leaf));
        }

        output.close();
        std::cout << "Converted vocabulary saved to: " << output_file << std::endl;
        std::cout << "Conversion completed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error during conversion: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}