#include "multi_layer_perceptron.hpp"
#include "letters_data_reader.hpp"
#include "mnist_data_reader.hpp"
#include <iostream>
#include <algorithm>
#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>

namespace {
void ensure_non_empty_dataset(
  const std::vector<std::vector<double> > &dataset,
  const std::vector<double> &labels,
  const std::string &dataset_name,
  const std::string &hint
) {
  if (dataset.empty() || labels.empty()) {
    throw std::runtime_error("Missing or empty " + dataset_name + " dataset. " + hint);
  }
  if (dataset.size() != labels.size()) {
    throw std::runtime_error("Mismatched " + dataset_name + " dataset and label sizes");
  }
}

std::string parse_backend(int argc, char **argv) {
  for (int index = 1; index < argc - 1; ++index) {
    if (std::string(argv[index]) == "--backend") {
      return argv[index + 1];
    }
  }
  return "cpu";
}

std::string parse_task_type(int argc, char **argv) {
  for (int index = 1; index < argc; ++index) {
    std::string argument = argv[index];
    if (argument == "--backend") {
      ++index;
      continue;
    }
    return argument;
  }
  return "letter-data";
}

void configure_backend(MultiLayerPerceptron &mlp, const std::string &backend_name) {
  std::string error_message;
  if (!mlp.set_execution_backend(backend_name, &error_message)) {
    throw std::runtime_error(error_message);
  }
  std::cout << "Using backend: " << mlp.get_execution_backend() << std::endl;
}
}

auto split_dataset(std::vector< std::vector<double> > dataset, std::vector<double> labels, double training_set_proportion) {
  std::vector<int> indexes(dataset.size(), 0);
  for (int i = 0; i < dataset.size(); ++i) {
    indexes[i] = i;
  }
  std::vector< std::vector<double >> training_set(0);
  std::vector<double> training_labels(0);
  std::vector< std::vector<double >> test_set(0);
  std::vector<double> test_labels(0);
  // shuffle indexes
  std::random_device rng;
  std::shuffle(indexes.begin(), indexes.end(), rng);
  int idx;
  for (idx = 0; idx < dataset.size() * training_set_proportion; ++idx) {
    training_set.push_back(dataset[indexes[idx]]);
    training_labels.push_back(labels[indexes[idx]]);
  }
  for (; idx < dataset.size(); ++idx) {
    test_set.push_back(dataset[indexes[idx]]);
    test_labels.push_back(labels[indexes[idx]]);
  }

  return std::make_tuple(training_set, training_labels, test_set, test_labels);
}

void executeLetterData(const std::string &backend_name) {
  LettersDataCSVReader reader = LettersDataCSVReader();
  auto dataset_and_labels = reader.read_letters_data();
  std::vector< std::vector<double> > dataset = std::get<0>(dataset_and_labels);
  std::vector<double> labels = std::get<1>(dataset_and_labels);
  ensure_non_empty_dataset(dataset, labels, "letter recognition", "Expected ./letter-recognition.data to be present.");
  int hidden_neurons[] = { 208, 104, 52 };
  MultiLayerPerceptron mlp = MultiLayerPerceptron(hidden_neurons, 2, 26, 10, 0.2, 0.9, 3, 32, "relu");
  configure_backend(mlp, backend_name);
  std::cout<<"Initializing"<<std::endl;
  auto splitted_dataset_tuple = split_dataset(dataset, labels, 0.8);
  std::vector< std::vector<double >> training_set = std::get<0>(splitted_dataset_tuple);
  std::vector<double> training_labels = std::get<1>(splitted_dataset_tuple);
  std::vector< std::vector<double >> test_set = std::get<2>(splitted_dataset_tuple);
  std::vector<double> test_labels = std::get<3>(splitted_dataset_tuple);

  mlp.train(training_set, training_labels);

  int training_correct_predictions = 0;
  for (int i = 0; i < training_set.size(); ++i) {
    double prediction = mlp.predict(training_set[i]);
    if (prediction == training_labels[i]) {
      training_correct_predictions++; 
    }
  }

  int test_correct_predictions = 0;
  for (int i = 0; i < test_set.size(); ++i) {
    double prediction = mlp.predict(test_set[i]);
    if (prediction == test_labels[i]) {
      test_correct_predictions++; 
    }
  }
  std::cout<<"\nAccuracy on training set is: "<<(double) training_correct_predictions / training_set.size()<<std::endl;
  std::cout<<"\nAccuracy on test set is: "<<(double) test_correct_predictions / test_set.size()<<std::endl;

}
const std::string DEFAULT_ASSETS = "./data/";
void executeMnist(const std::string &backend_name) {
  auto training_set = mnist::read_mnist_images_from_path(DEFAULT_ASSETS + "train-images-idx3-ubyte");
  auto training_labels = mnist::read_mnist_labels_from_path(DEFAULT_ASSETS + "train-labels-idx1-ubyte");
  auto test_set = mnist::read_mnist_images_from_path(DEFAULT_ASSETS + "t10k-images-idx3-ubyte");
  auto test_labels = mnist::read_mnist_labels_from_path(DEFAULT_ASSETS + "t10k-labels-idx1-ubyte");
  ensure_non_empty_dataset(training_set, training_labels, "MNIST training", "Expected train-images-idx3-ubyte and train-labels-idx1-ubyte in the project root.");
  ensure_non_empty_dataset(test_set, test_labels, "MNIST test", "Expected t10k-images-idx3-ubyte and t10k-labels-idx1-ubyte in the project root.");

  std::cout<<" X " << training_labels[0]<<std::endl;
  int hidden_neurons[] = { 208, 104, 52 };
  MultiLayerPerceptron mlp = MultiLayerPerceptron(hidden_neurons, 4, 10, 10, 0.1, 0.9, 3, 32, "relu");
  configure_backend(mlp, backend_name);
  mlp.train(training_set, training_labels);

  int training_correct_predictions = 0;
  for (int i = 0; i < training_set.size(); ++i) {
    double prediction = mlp.predict(training_set[i]);
    if (prediction == training_labels[i]) {
      training_correct_predictions++; 
    }
  }

  std::cout<<" SIZE " << test_set.size()<< " " << test_labels.size() << std::endl;
  int test_correct_predictions = 0;
  for (int i = 0; i < test_set.size(); ++i) {
    double prediction = mlp.predict(test_set[i]);
    if (prediction == test_labels[i]) {
      test_correct_predictions++;
    }
  }

  std::cout<<"\nAccuracy on training set is: "<<(double) training_correct_predictions / training_set.size()<<std::endl;
  std::cout<<"\nAccuracy on test set is: "<<(double) test_correct_predictions / test_set.size()<<std::endl;
}

int main(int argc, char **argv) {
  try {
    const std::string backend_name = parse_backend(argc, argv);
    const std::string task_type = parse_task_type(argc, argv);
    if (task_type == "letter-data") {
      executeLetterData(backend_name);
    } else if (task_type == "mnist") {
      executeMnist(backend_name);
    } else {
      std::cerr << "Usage: ./mlp [letter-data|mnist] [--backend cpu|gpu]" << std::endl;
      return -1;
    }
  } catch (const std::exception &exception) {
    std::cerr << exception.what() << std::endl;
    return -1;
  }
  return 0;
}
