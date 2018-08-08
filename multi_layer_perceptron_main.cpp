#include "multi_layer_perceptron.hpp"
#include "letters_data_reader.hpp"
#include <iostream>
#include <algorithm>
#include <vector>
#include <random>
#include <cmath>

auto split_dataset(std::vector< std::vector<double> > dataset, std::vector<double> labels, double training_set_proportion) {
  std::vector<int> indexes;
  indexes.reserve(dataset.size());
  for (int i = 0; i < dataset.size(); ++i) {
    indexes[i] = i;
  }
  std::vector< std::vector<double >> training_set(0);
  std::vector<double> training_labels(0);
  std::vector< std::vector<double >> test_set(0);
  std::vector<double> test_labels(0);
  // shuffle indexes
  std::random_shuffle(indexes.begin(), indexes.end());
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

void executeLetterData() {
  LettersDataCSVReader reader = LettersDataCSVReader();
  auto dataset_and_labels = reader.read_letters_data();
  std::vector< std::vector<double> > dataset = std::get<0>(dataset_and_labels);
  std::vector<double> labels = std::get<1>(dataset_and_labels);
  MultiLayerPerceptron mlp = MultiLayerPerceptron(128, 26, 1000, 0.001, "sigmoidal");

  auto splitted_dataset_tuple = split_dataset(dataset, labels, 0.8);
  std::vector< std::vector<double >> training_set = std::get<0>(splitted_dataset_tuple);
  std::vector<double> training_labels = std::get<1>(splitted_dataset_tuple);
  std::vector< std::vector<double >> test_set = std::get<2>(splitted_dataset_tuple);
  std::vector<double> test_labels = std::get<3>(splitted_dataset_tuple);
 
  mlp.train(training_set, training_labels);
  
  unsigned int training_correct_predictions = 0;
  for (int i = 0; i < test_set.size(); ++i) {
    double prediction = mlp.predict(training_set[i]);
    if (prediction == training_labels[i]) {
      training_correct_predictions++; 
    }
  }
  std::cout<<"\nAccuracy on training set is: "<<(double) training_correct_predictions / test_set.size();
  
  unsigned int test_correct_predictions = 0;
  for (int i = 0; i < test_set.size(); ++i) {
    double prediction = mlp.predict(test_set[i]);
    if (prediction == test_labels[i]) {
      test_correct_predictions++; 
    }
  }
  std::cout<<"\nAccuracy on test set is: "<<(double) test_correct_predictions / test_set.size();

}

int main(int argc, char **argv) {
  if (argc <= 1) {
    return 1;
  } else {
    std::string task_type = argv[1];
    if (task_type == "letter-data") {
     executeLetterData(); 
    } else {
      return -1;
    }
  }
}
