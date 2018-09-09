#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mnist {
  int from_byte_to_int(int number_byte) {
    unsigned char c1, c2, c3, c4;
    c1 = number_byte & 255;
    c2 = (number_byte >> 8) & 255;
    c3 = (number_byte >> 16) & 255;
    c4 = (number_byte >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;		
  }

  std::pair<int, int> read_magic_number_and_size(std::ifstream &mnist_file, int magic_number_match) {
    int magic_number, dataset_size;
    mnist_file.read((char *) &magic_number, sizeof(magic_number));
    magic_number = from_byte_to_int(magic_number);
    if(magic_number != magic_number_match) {
      throw std::runtime_error("Invalid MNIST image mnist_file!");
    }

    mnist_file.read((char *) &dataset_size, sizeof(dataset_size)), dataset_size = from_byte_to_int(dataset_size);
    return std::make_pair(magic_number, dataset_size);
  }

  // change to template on mlp
  std::vector<std::vector<double> > read_mnist_images_from_path(std::string path) {
    std::ifstream mnist_file(path, std::ios::binary);
    std::vector<std::vector<double> > images;
    if (mnist_file.is_open()) {
      int image_rows_size = 0;
      int image_cols_size = 0;

      std::pair<int, int> magic_number_and_size = read_magic_number_and_size(mnist_file, 2051);
      int magic_number = magic_number_and_size.first;
      int dataset_size = magic_number_and_size.second;

      mnist_file.read((char *) &image_rows_size, sizeof(image_rows_size)), image_rows_size = from_byte_to_int(image_rows_size);
      mnist_file.read((char *) &image_cols_size, sizeof(image_cols_size)), image_cols_size = from_byte_to_int(image_cols_size);
      int image_size = image_cols_size * image_rows_size;
      images = std::vector< std::vector<double> >(dataset_size);
      for (int j = 0; j < dataset_size; ++j) {
        images[j] = std::vector<double>(image_size);
        for (int i = 0; i < image_size; ++i) {
          uint8_t value;
          mnist_file.read((char *) &value, sizeof(value));
          images[j][i] = value / 255.0;
        }
      }
    }
    return images;
  }

  std::vector<double> read_mnist_labels_from_path(std::string path) {
    std::ifstream mnist_file(path, std::ios::binary);
    std::vector<double> labels;
    if (mnist_file.is_open()) {
      std::pair<int, int> magic_number_and_size = read_magic_number_and_size(mnist_file, 2049);
      int magic_number = magic_number_and_size.first;
      int dataset_size = magic_number_and_size.second;
      labels = std::vector<double>(dataset_size);
      for (int j = 0; j < dataset_size; ++j) {
        uint8_t value;
        mnist_file.read((char *) &value, sizeof(unsigned char));
        labels[j] = (double) value;
      }
    }
    return labels;
  }
}

/*
   int main() {
// mnist::read_mnist_images_from_path("train-images-idx3-ubyte");
mnist::read_mnist_labels_from_path("train-labels-idx1-ubyte");
}
*/
