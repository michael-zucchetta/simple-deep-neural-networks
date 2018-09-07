#ifndef MNIST_DATA_READER_HPP
#define MNIST_DATA_READER_HPP
#include <string>

namespace mnist {
  std::vector<std::vector<double> > read_mnist_images_from_path(std::string path);
  std::vector<double> read_mnist_labels_from_path(std::string path);
}
#endif
