#include "multi_layer_perceptron.hpp"
#include "letters_data_reader.hpp"
#include "mnist_data_reader.hpp"
#include <iostream>
#include <algorithm>
#include <vector>
#include <random>
#include <cmath>

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

void executeLetterData() {
	LettersDataCSVReader reader = LettersDataCSVReader();
	auto dataset_and_labels = reader.read_letters_data();
	std::vector< std::vector<double> > dataset = std::get<0>(dataset_and_labels);
	std::vector<double> labels = std::get<1>(dataset_and_labels);
	int hidden_neurons[] = { 208, 104, 52 };
	MultiLayerPerceptron mlp = MultiLayerPerceptron(hidden_neurons, 1, 26, 100, 0.01, 0.9, 3, 32, "relu");
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

void executeMnist() {
	auto training_set = mnist::read_mnist_images_from_path("train-images-idx3-ubyte");
	auto training_labels = mnist::read_mnist_labels_from_path("train-labels-idx1-ubyte");
	auto test_set = mnist::read_mnist_images_from_path("t10k-images-idx3-ubyte");
	auto test_labels = mnist::read_mnist_labels_from_path("t10l-labels-idx1-ubyte");

	std::cout<<" X " << training_labels[0]<<std::endl;
	int hidden_neurons[] = { 208, 104, 52 };
	MultiLayerPerceptron mlp = MultiLayerPerceptron(hidden_neurons, 1, 26, 100, 0.1, 0.9, 3, 32, "relu");
	mlp.train(training_set, training_labels);

	int training_correct_predictions = 0;
	for (int i = 0; i < training_set.size(); ++i) {
		double prediction = mlp.predict(training_set[i]);
		if (prediction == training_labels[i]) {
			training_correct_predictions++; 
		}
	}

	std::cout<<"\nAccuracy on training set is: "<<(double) training_correct_predictions / training_set.size()<<std::endl;
}

int main(int argc, char **argv) {
	if (argc <= 1) {
		executeLetterData();
		return 1;
	} else {
		std::string task_type = argv[1];
		if (task_type == "letter-data") {
			executeLetterData();
		} else if (task_type == "mnist") {
			executeMnist();
		} else {
			return -1;
		}
	}
}
