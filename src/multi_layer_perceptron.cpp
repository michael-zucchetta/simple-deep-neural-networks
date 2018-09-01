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
static std::uniform_real_distribution<> dis(-0.05, 0.05);

double get_random() {
  double random_number = dis(e) + 0.00000001;
  if (random_number == 0) {
    return 0.0000001;
  } else {
    return random_number;
  }
}

std::vector<double> init_random_values(int size) {
  std::vector<double> tmp_vector(size);
  for (int i = 0; i < size; ++i) {
    tmp_vector[i] = get_random();
  }
  return tmp_vector;
}

std::vector< std::vector<double> > init_weights(const int neurons_size, const int size_of_each_vector, bool is_derivative) {
  std::vector< std::vector<double> > tmp_vector(neurons_size);
  for (int i = 0; i < neurons_size; ++i) {
    if (is_derivative) {
      tmp_vector[i] = std::vector<double>(size_of_each_vector, 0.0);
    } else {
      tmp_vector[i] = init_random_values(size_of_each_vector);
    }
  }
  return tmp_vector;
}

void MultiLayerPerceptron::initialize(int inputs_size) {
  this->weights = std::vector< std::vector<std::vector<double> > >(0);
  this->weight_derivatives = std::vector< std::vector<std::vector<double> > >(0);
  this->prev_velocities = std::vector< std::vector<std::vector<double> > >(0);
  
  std::vector<int> neuron_sizes = std::vector<int>(0);
  neuron_sizes.push_back(inputs_size);
  for (int i = 0; i < layers; ++i) {
    neuron_sizes.push_back(this->hidden_neurons[i]);
  }
  neuron_sizes.push_back(this->size_outputs);
  for (int i = 1; i < neuron_sizes.size(); ++i) {
    auto weights_layer = init_weights(neuron_sizes[i], neuron_sizes[i - 1], false);
    this->weights.push_back(weights_layer);
    auto derivative_weights_layer = init_weights(neuron_sizes[i], neuron_sizes[i - 1], true);
    this->weight_derivatives.push_back(derivative_weights_layer);
    auto prev_derivative_weights_layer = init_weights(neuron_sizes[i], neuron_sizes[i - 1], true);
    this->prev_velocities.push_back(prev_derivative_weights_layer);
  }
}

double sigmoid(double value) {
  return 1.0 / ( 1.0 + exp(-value));
}

double derivative_sigmoid(double value) {
  return sigmoid(value) * ( 1.0 - sigmoid(value));
}

std::vector<double> activation(std::vector<double> values, const std::string activation) {
  std::vector<double> activated_values(values.size());
  double sum = 0.0;
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
      // value = log(1.0 + exp(values[i]));
    } else if (activation == "linear") {
      value = values[i];
    } else if (activation == "softmax") {
      value = exp(values[i]);
      sum += value;
    }

    activated_values[i] = value;
  }
  if (activation == "softmax") {
    for (int i = 0; i < values.size(); ++i) {
      activated_values[i] /= sum;
    }
  }
  return activated_values;
}

std::vector<double> derivative_activation(std::vector<double> values, std::string activation) {
  std::vector<double> activated_values(values.size());
  double sum = 0.0;
  for (int i = 0; i < values.size(); ++i) {
    double value;
    if (activation == "sigmoidal") {
      value = derivative_sigmoid(values[i]);
    } else if (activation == "linear") {
      value = 1.0; 
    } else if (activation == "relu") {
      if (values[i] < 0) {
        value = 0.0;
      } else {
        value = 1.0;//value;//1.0;
      }
      // value = 1.0 / (1.0 + exp(-values[i]));
    } else if (activation == "softmax") {
      value = exp(values[i]);
      sum += value;
    }
    activated_values[i] = value;
  }
  if (activation == "softmax") {
    for (int i = 0; i < values.size(); ++i) {
      activated_values[i] /= sum;
    }
  }
  return activated_values;
}

// the real init is done in the first training as the MultiLayerPerceptron needs to know the size of the inputs and outputs for initializing
// the weights
MultiLayerPerceptron::MultiLayerPerceptron(int _hidden_neurons[], int _size_outputs, int _epochs, float _learning_rate, float _momentum, int _layers, std::string _activation_function) {
  srand((int) time (NULL)); // generator
  size_outputs = _size_outputs;
  hidden_neurons = _hidden_neurons;
  epochs = _epochs;
  learning_rate = _learning_rate;
  momentum = _momentum;
  first_time_training = true;
  layers = _layers;
  activation_function = _activation_function;
  this->zetas = std::vector<std::vector<double> >(layers + 1);
  this->stored_activated_values = std::vector<std::vector<double> >(layers + 1);
}

std::vector<double> MultiLayerPerceptron::forward_propagation(std::vector<double> X) {
  for (int i = 0; i < this->weights.size(); i++) {
    std::vector<double> zeta(this->weights[i].size());
    std::vector<double> activated_weights;
    if (i != 0) {
      activated_weights = activation(this->zetas[i - 1], this->activation_function);
      this->stored_activated_values[i - 1] = activated_weights;
    }
    for (int y = 0; y < this->weights[i].size(); ++y) {
      double tmp_z = 0.0;
      for (int x = 0; x < this->weights[i][y].size(); ++x) {
        if (i == 0) {
          tmp_z += this->weights[i][y][x] * X[x]; 
        } else { 
          tmp_z += this->weights[i][y][x] * activated_weights[x];
        }
      }
      zeta[y] = tmp_z;
    }
    this->zetas[i] = zeta;
  }
  auto outputs = activation(this->zetas[this->layers], "softmax");
  this->stored_activated_values[this->layers] = outputs;
  return outputs; 
}

double MultiLayerPerceptron::back_propagation(std::vector<double> X, std::vector<double> targets, std::vector<double> outputs) {
  std::vector<double> delta_output(outputs.size());
  for (int i = 0; i < outputs.size(); ++i) {
    // delta_output[i] = targets[i] - outputs[i];
    delta_output[i] = outputs[i] - targets[i];
  }
  std::vector<double> d_z(delta_output.size());
  // std::vector<double> der_sigm_z_2 = derivative_activation(outputs, activation_function);
  /*for (int y = 0; y < delta_output.size(); ++y) {
      d_z[y] = delta_output[y];// * der_sigm_z_2[y];
  }*/
  d_z = delta_output;//derivative_activation(delta_output, "sigmoidal");
  for (int i = this->layers - 1; i >= 0; i--) {
    std::vector<double> der_sigm_z = derivative_activation(this->zetas[i], this->activation_function);
    std::vector<double> prev_d_z = d_z;
    d_z = std::vector<double>(hidden_neurons[i], 0.0);
    for (int y = 0; y < this->weights[i + 1].size(); ++y) {
      for (int x = 0; x < this->weights[i + 1][y].size(); ++x) {
        d_z[x] += this->weights[i + 1][y][x] * prev_d_z[y] * der_sigm_z[x];
      }
    }
    for (int y = 0; y < prev_d_z.size(); ++y) {
      for (int x = 0; x < this->stored_activated_values[i].size(); ++x) {
        this->weight_derivatives[i + 1][y][x] += prev_d_z[y] * this->stored_activated_values[i][x];
        //if (this->weight_derivatives[i + 1][y][x] == 0.0) {
        //}
      }
    }
  }
  for (int y = 0; y < d_z.size(); ++y) {
    for (int x = 0; x < X.size(); ++x) {
      this->weight_derivatives[0][y][x] += d_z[y] * X[x];
    }
  }

  double error = 0.0;
  for (int i = 0; i < delta_output.size(); ++i) {
    // squared error
    error += std::abs(delta_output[i]);
  }
  return pow(error, 2);
}

void MultiLayerPerceptron::update_weights(const int actual_batch_size) {

	for (int i = 0; i < this->weights.size(); ++i) {
    for (int y = 0; y < this->weights[i].size(); ++y) {
      for (int x = 0; x < this->weights[i][y].size(); ++x) {
	 // Gradient descent
         // std::cout<< this->prev_velocities[i][y][x]<< " " << average_weight_derivative << " FF 1: " <<(average_weight_derivative - this->prev_velocities[i][y][x] / (double)  actual_batch_size)<<std::endl;

	 double velocity_momentum = this->momentum * this->prev_velocities[i][y][x] + (1 - this->momentum) * learning_rate * this->weight_derivatives[i][y][x];
	 double velocity_nesterov = velocity_momentum + this->momentum * (velocity_momentum - this->prev_velocities[i][y][x]);
	 // MOMENTUM double velocity = this->momentum * this->prev_velocities[i][y][x] + (1 - this->momentum) * learning_rate * this->weight_derivatives[i][y][x];
	this->weights[i][y][x] = this->weights[i][y][x] - velocity_nesterov;
        this->prev_velocities[i][y][x] = velocity_nesterov;
        this->weight_derivatives[i][y][x] = 0.0;
      }
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
    this->initialize(inputs_size);
    first_time_training = false;
  }
  std::cout<<"Start training"<<std::endl;
  std::vector< std::vector<double> > new_labels(labels.size());
  // transform labels
  std::transform(labels.begin(), labels.end(), new_labels.begin(), [this] (double label) {
      return transform_label(label);
  });
  predict(training_set[0]);
  int did_not_decrease = 0;
  auto rng = std::default_random_engine {};
  
  std::vector<int> indexes(training_set.size());
  for (int i = 0; i < training_set.size(); ++i) {
    indexes[i] = i;
  }
  this->batch_size = training_set.size() > 32? 32 : training_set.size();
  int batch_count = 0;
  for (int i = 0; i < epochs; ++i) {
    clock_t begin = clock();
    double loss = 0.0;
    // shuffle indexes
    std::shuffle(indexes.begin(), indexes.end(), r);
    for (int ind = 0; ind < training_set.size(); ++ind) {
      std::vector<double> outputs = this->forward_propagation(training_set[indexes[ind]]);
      loss += this->back_propagation(training_set[indexes[ind]], new_labels[indexes[ind]], outputs);
      batch_count++;
      auto module_batch_count_size = (batch_count) % this->batch_size;
      if (module_batch_count_size == 0 || ind + 1 == training_set.size()) {
        auto actual_batch_size = 0;
	if (module_batch_count_size == 0) {
	  actual_batch_size = this->batch_size;
	} else {
	  actual_batch_size = training_set.size() % this->batch_size;
	}
	update_weights(actual_batch_size);
        batch_count = 0;
      }
    }
    std::cout<<"Epoch n. "<<(i + 1)<<", with error "<<loss/(double) training_set.size()<<std::endl;
    clock_t end = clock();
    std::cout<<"Time for single epoch is "<< double(end - begin) / CLOCKS_PER_SEC <<std::endl;
  }
}

double MultiLayerPerceptron::predict(std::vector<double> item) {
	this->forward_propagation(item);
  if (this->zetas[this->layers].size() == 1) {
    if (activation_function == "sigmoidal") { 
      std::cout<<"Value predicted, before returning actual class: "<<this->zetas[this->layers][0]<<std::endl;
      if (this->zetas[this->layers + 1][0] > 0.5) {
		    return 1;
	    } else {
		    return 0;
	    }
    } else {
      // else not classification
      return this->zetas[this->layers][0];
    }
  } else {
    int predicted_output = -1;
    double predicted_value = -1;
    for (int i = 0; i < this->zetas[this->layers].size(); ++i) {
      if (predicted_value < this->zetas[this->layers][i]) {
        predicted_value = this->zetas[this->layers][i];
        predicted_output = i;
      }
    }
    return predicted_output;
  }
}
