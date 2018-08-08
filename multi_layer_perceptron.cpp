#include <iostream>
#include <ctime>
#include <tuple>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <limits>
#include <set>
#include "multi_layer_perceptron.hpp"

static std::random_device r; // seed
std::default_random_engine e{r()};
static std::uniform_real_distribution<> dis(0.0001, 0.1);

double get_random() {
  double random_number = dis(e);
  return random_number;
}

std::vector<double> init_random_values(unsigned int size) {
  std::vector<double> tmp_vector(size);
  for (int i = 0; i < size; ++i) {
    tmp_vector[i] = get_random();
  }
  return tmp_vector;
}

std::vector< std::vector<double> > init_weights(unsigned int number_of_vectors, unsigned int size_of_each_vector) {
  std::vector< std::vector<double> > tmp_vector(number_of_vectors);
  for (int i = 0; i < number_of_vectors; ++i) {
    tmp_vector[i] = init_random_values(size_of_each_vector);
  }
  return tmp_vector;
}

void MultiLayerPerceptron::randomize(unsigned int inputs_size) {
  w1 = init_weights(hidden_neurons, inputs_size);
  dw1 = std::vector< std::vector<double> >(0); 
  for (int i = 0; i < w1.size(); ++i) {
    std::vector<double> tmp(0);
    for (int j = 0; j < w1[i].size(); ++j) {
      tmp.push_back(0.0);
    }
    dw1.push_back(tmp);
  }
  w2 = init_weights(size_outputs, hidden_neurons);
  dw2 = std::vector< std::vector<double> >(0); 
  for (int i = 0; i < w2.size(); ++i) {
    std::vector<double> tmp(0);
    for (int j = 0; j < w2[i].size(); ++j) {
      tmp.push_back(0.0);
    }
    dw2.push_back(tmp);
  }
}

double sigmoid(double value) {
  return 1.0 / ( 1.0 + exp(-value));
}

double derivative_sigmoid(double value) {
  return sigmoid(value) * ( 1.0 - sigmoid(value));
}

std::vector<double> activation(std::vector<double> values, std::string activation) {
  std::vector<double> activated_values(0);
  for (int i = 0; i < values.size(); ++i) {
    double value;
    if (activation == "sigmoidal") {
      value = sigmoid(values[i]);
    } else if (activation == "relu") {
      if (values[i] > 0) {
        value = values[i];
      } else {
        value = 0;
      }
    } else if (activation == "linear") {
      value = values[i];
    }
    activated_values.push_back(value);
  }
  return activated_values;
}

std::vector<double> derivative_activation(std::vector<double> values, std::string activation) {
  std::vector<double> activated_values(0);
  for (int i = 0; i < values.size(); ++i) {
    double value;
    if (activation == "sigmoidal") {
      value = derivative_sigmoid(values[i]);
    } else if (activation == "linear") {
      value = 1.0; 
    } else if (activation == "relu") {
      value = 1.0;//values[i];
    }
    activated_values.push_back(value);
  }
  return activated_values;
}

// the real init is done in the first training as the MultiLayerPerceptron needs to know the size of the inputs and outputs for initializing
// the weights
MultiLayerPerceptron::MultiLayerPerceptron(double _hidden_neurons, double _size_outputs, int _epochs, double _learning_rate, std::string _activation_function) {
  srand((unsigned int) time (NULL)); // generator
  size_outputs = _size_outputs;
  hidden_neurons = _hidden_neurons;
  epochs = _epochs;
  learning_rate = _learning_rate;
  first_time_training = true;
  activation_function = _activation_function;
}

void MultiLayerPerceptron::forward_propagation(std::vector<double> X) {
  std::vector<double> _z_1(0); 
  for (int y = 0; y < w1.size(); ++y) {
    double tmp_z_1 = 0.0;
    for (int x = 0; x < w1[y].size(); ++x) {
      tmp_z_1 += w1[y][x] * X[x];
    }
    _z_1.push_back(tmp_z_1);
  }
	z1 = _z_1;
  // hidden units are always sigmoidal
  h = activation(z1, activation_function);
  
  std::vector<double> _z_2(0); 
 
  // refactoring 
  for (int y = 0; y < w2.size(); ++y) {
    double tmp_z_2 = 0.0;
    for (int x = 0; x < w2[y].size(); ++x) {
      tmp_z_2 += w2[y][x] * h[x];
    }
    _z_2.push_back(tmp_z_2);
  }
	z2 = _z_2;

  o = activation(z2, "sigmoidal"); 
}

double MultiLayerPerceptron::back_propagation(std::vector<double> X, std::vector<double> targets) {
  std::vector<double> delta_output(0);
  for (int i = 0; i < o.size(); ++i) {
    delta_output.push_back(targets[i] - o[i]); 
  }

  std::vector<double> d_z_2(0);
  std::vector<double> der_sigm_z_2 = derivative_activation(z2, activation_function);
  for (int y = 0; y < delta_output.size(); ++y) {
      d_z_2.push_back(delta_output[y]);// * der_sigm_z_2[y]);
  }

  for (int y = 0; y < delta_output.size(); ++y) {
    for (int x = 0; x < h.size(); ++x) {
      dw2[y][x] += d_z_2[y] * h[x];
    }
  }

  std::vector<double> der_sigm_z_1 = derivative_activation(z1, activation_function); 
  // std::vector<double> der_sigm_z_1 = derivative_activation(z1, activation_function); 
  std::vector<double> d_z_1(hidden_neurons);
  for (int y = 0; y < w2.size(); ++y) {
    for (int x = 0; x < w2[y].size(); ++x) {
      d_z_1[x] += w2[y][x] * d_z_2[y] * der_sigm_z_1[x];
    }
  }

  for (int y = 0; y < d_z_1.size(); ++y) {
    for (int x = 0; x < X.size(); ++x) {
      dw1[y][x] += d_z_1[y] * X[x];
    }
  }
  double error = 0.0;
  for (int i = 0; i < delta_output.size(); ++i) {
    // squared error
    error += pow(delta_output[i], 2);
  }
  return error; 
}

void MultiLayerPerceptron::update_weights() {
	for (int y = 0; y < w1.size(); ++y) {
    for (int x = 0; x < w1[y].size(); ++x) {
      w1[y][x] = w1[y][x] + learning_rate * dw1[y][x];
      dw1[y][x] = 0.0;
    }
  }
	for (int y = 0; y < w2.size(); ++y) {
    for (int x = 0; x < w2[y].size(); ++x) {
      w2[y][x] = w2[y][x] + learning_rate * dw2[y][x];
      dw2[y][x] = 0.0;
    }
  }
}

/*
 * Transform the single label to an vector of size equals to the number of outputs
 */
std::vector<double> MultiLayerPerceptron::transform_label(double label) {
  if (size_outputs > 1 && size_outputs >= labels_unique_size) { 
    std::vector<double> tmp_vector(size_outputs, 0);
    for (int i = 0; i < labels_unique_size; ++i) {
      if (i == label) { 
        // assumption that the labels starts from zero and have sequential, contiguos values
        // this means that other kind of data require pre-processing 
        tmp_vector[i] = 1;
      }
      // everything else is already initialized as zero
    }
    return tmp_vector;
  } else {
    /*
     * It might be possible to variate the labels with the number of outputs
     * but it might be quite unpredictive
     */
    std::vector<double> tmp_vector(size_outputs, 0.0);
    for (int i = 0; i < size_outputs; ++i) {
      tmp_vector[i] = label;
    }
    return tmp_vector; 
  } 
}

void MultiLayerPerceptron::train(std::vector< std::vector<double> > training_set, std::vector<double> labels) {
	std::cout.precision(17);
  if (first_time_training) {
    std::cout<<"\nTraining size: "<<training_set.size()<<" with n. "<<epochs<<" epochs"<<std::endl;
    int inputs_size = training_set[0].size(); // must be same length as labels
    labels_unique_size = std::set<double>( labels.begin(), labels.end() ).size(); 
    w1 = init_weights(hidden_neurons, inputs_size);
    dw1 = std::vector< std::vector<double> >(0); 
    for (int i = 0; i < w1.size(); ++i) {
      std::vector<double> tmp(0);
      for (int j = 0; j < w1[i].size(); ++j) {
        tmp.push_back(0.0);
      }
      dw1.push_back(tmp);
    }
		w2 = init_weights(size_outputs, hidden_neurons);
    dw2 = std::vector< std::vector<double> >(0); 
		for (int i = 0; i < w2.size(); ++i) {
      std::vector<double> tmp(0);
      for (int j = 0; j < w2[i].size(); ++j) {
        tmp.push_back(0.0);
      }
      dw2.push_back(tmp);
    }

    first_time_training = false;
	
  }
  std::vector< std::vector<double> > new_labels(labels.size());
  // transform labels
  std::transform(labels.begin(), labels.end(), new_labels.begin(), [this] (double label) {
      return transform_label(label);
  });
  predict(training_set[0]);
	int did_not_decrease = 0;
	auto rng = std::default_random_engine {};
  
  std::vector<int> indexes;
  indexes.reserve(training_set.size());
  for (int i = 0; i < training_set.size(); ++i) {
    indexes.push_back(i);
  }
  int batch_size = training_set.size() > 200? 200 : training_set.size();
  int batch_count = 0;
  for (int i = 0; i < epochs; ++i) {
		double loss = 0.0;
    // shuffle indexes
    std::random_shuffle(indexes.begin(), indexes.end());
    for (int ind = 0; ind < training_set.size(); ++ind) {
			forward_propagation(training_set[indexes[ind]]);
      loss += back_propagation(training_set[indexes[ind]], new_labels[indexes[ind]]);
      batch_count++;
      if ((batch_count) % batch_size == 0) {
        update_weights();
        batch_count = 0;
      }
    }
    std::cout<<"Epoch n. "<<(i + 1)<<", with error "<<loss/(double) training_set.size()<<std::endl;
  }
}

double MultiLayerPerceptron::predict(std::vector<double> item) {
	forward_propagation(item);
	if (o.size() == 1) {
    if (activation_function == "sigmoidal") { 
      std::cout<<"Value predicted, before returning actual class: "<<o[0]<<std::endl;
      if (o[0] > 0.5) {
		    return 1;
	    } else {
		    return 0;
	    }
    } else {
      // else not classification
      return o[0];
    }
  } else {
    int predicted_output = -1;
    double predicted_value = -1;
    for (int i = 0; i < o.size(); ++i) {
      if (predicted_value < o[i]) {
        predicted_value = o[i];
        predicted_output = i;
      }
    }
    return predicted_output;
  }
}
