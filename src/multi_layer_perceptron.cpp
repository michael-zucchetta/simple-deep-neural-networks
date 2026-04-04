#include <iostream>
#include <ctime>
#include <tuple>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <limits>
#include <set>
#include <future>
#include <thread>
#include <chrono>
#include <stdexcept>
#include "multi_layer_perceptron.hpp"
#include "gpu_training_backend.hpp"

static std::random_device r; // seed
std::default_random_engine e{r()};
static std::uniform_real_distribution<> dis(-0.05, 0.05);
typedef std::chrono::system_clock Clock;

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

TrainingBatch make_training_batch(
  std::vector<std::vector<double> > &X,
  std::vector<int> &indexes,
  int index_from,
  int index_to,
  int output_size,
  const std::vector<std::vector<double> > *targets = nullptr
) {
  TrainingBatch batch;
  batch.batch_size = index_to - index_from;
  batch.input_size = batch.batch_size > 0 ? X[indexes[index_from]].size() : 0;
  batch.output_size = output_size;
  batch.inputs.reserve(batch.batch_size * batch.input_size);
  for (int z = index_from; z < index_to; ++z) {
    const std::vector<double> &sample = X[indexes[z]];
    batch.inputs.insert(batch.inputs.end(), sample.begin(), sample.end());
  }
  if (targets != nullptr) {
    batch.targets.reserve(batch.batch_size * output_size);
    for (int z = index_from; z < index_to; ++z) {
      const std::vector<double> &target = (*targets)[indexes[z]];
      batch.targets.insert(batch.targets.end(), target.begin(), target.end());
    }
  } else {
    batch.targets.assign(batch.batch_size * output_size, 0.0);
  }
  return batch;
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
  // to fix
  this->zetas = std::vector<std::vector<std::vector<double> > >(this->batch_size);
  this->stored_activated_values = std::vector<std::vector<std::vector<double> > >(this->batch_size);
  for (int i = 0; i < this->batch_size; ++i) {
    this->zetas[i] = std::vector<std::vector<double> >(layers + 1);
    this->stored_activated_values[i] = std::vector<std::vector<double> >(layers + 1);
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
MultiLayerPerceptron::MultiLayerPerceptron(int _hidden_neurons[], int _threads_size, int _size_outputs, int _epochs, float _learning_rate, float _momentum, int _layers, int _batch_size, std::string _activation_function) {
  srand((int) time (NULL)); // generator
  this->threads_size = _threads_size;
  this->hidden_neurons = _hidden_neurons;
  this->size_outputs = _size_outputs;
  this->epochs = _epochs;
  this->learning_rate = _learning_rate;
  this->momentum = _momentum;
  this->first_time_training = true;
  this->layers = _layers;
  this->batch_size = _batch_size;
  this->activation_function = _activation_function;
  this->execution_backend = "cpu";
  /*this->zetas = std::vector<std::vector<std::vector<double> > >(this->batch_size);
    this->stored_activated_values = std::vector<std::vector<std::vector<double> > >(this->batch_size);
    for (int i = 0; i < this->batch_size; ++i) {
    this->zetas[i] = std::vector<std::vector<double> >(layers + 1);
    this->stored_activated_values[i] = std::vector<std::vector<double> >(layers + 1);
    }*/
}

std::vector<std::vector<double> > MultiLayerPerceptron::forward_propagation(std::vector<std::vector<double> > &X, std::vector<int> &indexes, int index_from, int index_to, int thread_index) {
  std::vector<std::vector<double> > outputs_by_batch(X.size());
  if (this->uses_gpu_backend()) {
    if (!this->training_backend->synchronize_model(this->export_flat_model())) {
      throw std::runtime_error(this->training_backend->last_error().empty() ? "GPU backend failed to synchronize model" : this->training_backend->last_error());
    }

    TrainingBatch batch = make_training_batch(X, indexes, index_from, index_to, this->size_outputs);
    BatchTrainingResult result;
    if (!this->training_backend->forward_batch(batch, &result)) {
      throw std::runtime_error(this->training_backend->last_error().empty() ? "GPU backend failed during forward propagation" : this->training_backend->last_error());
    }

    const int batch_size = index_to - index_from;
    if (static_cast<int>(result.layer_zetas.size()) != this->layers + 1 || static_cast<int>(result.layer_activations.size()) != this->layers + 1) {
      throw std::runtime_error("GPU backend returned incomplete layer outputs");
    }

    for (int sample_index = 0; sample_index < batch_size; ++sample_index) {
      const int contextual_index = sample_index + thread_index;
      for (int layer_index = 0; layer_index < this->layers + 1; ++layer_index) {
        const LayerBatchValues &zetas = result.layer_zetas[layer_index];
        const LayerBatchValues &activations = result.layer_activations[layer_index];
        this->zetas[contextual_index][layer_index] = std::vector<double>(
          zetas.values.begin() + sample_index * zetas.value_size,
          zetas.values.begin() + (sample_index + 1) * zetas.value_size
        );
        this->stored_activated_values[contextual_index][layer_index] = std::vector<double>(
          activations.values.begin() + sample_index * activations.value_size,
          activations.values.begin() + (sample_index + 1) * activations.value_size
        );
      }

      outputs_by_batch[sample_index] = std::vector<double>(
        result.outputs.begin() + sample_index * this->size_outputs,
        result.outputs.begin() + (sample_index + 1) * this->size_outputs
      );
    }

    return outputs_by_batch;
  }

  for (int z = 0; z < index_to - index_from; ++z) {
    int z_contextual_idx = z + thread_index;
    for (int i = 0; i < this->weights.size(); ++i) {
      auto zeta = std::vector<double>(this->weights[i].size());
      std::vector<double> layer_input;
      if (i == 0) {
        layer_input = X[indexes[index_from + z]];
      } else {
        layer_input = activation(this->zetas[z_contextual_idx][i - 1], this->activation_function);
        this->stored_activated_values[z_contextual_idx][i - 1] = layer_input;
      }

      for (int y = 0; y < this->weights[i].size(); ++y) {
        double tmp_z = 0.0;
        for (int x = 0; x < this->weights[i][y].size(); ++x) {
          tmp_z += this->weights[i][y][x] * layer_input[x];
        }
        zeta[y] = tmp_z;
      }
      this->zetas[z_contextual_idx][i] = zeta;
    }
    auto outputs = activation(this->zetas[z_contextual_idx][this->layers], "softmax");
    this->stored_activated_values[z_contextual_idx][this->layers] = outputs;
    outputs_by_batch[z] = outputs;
  }
  return outputs_by_batch;
}

double MultiLayerPerceptron::back_propagation(std::vector<std::vector<double> > &X, std::vector<std::vector<double> > &targets, std::vector<std::vector<double> > &outputs, std::vector<int> &indexes, int index_from, int index_to, int thread_index) {
  double error = 0.0;
  for (int z = 0; z < index_to - index_from; ++z) {
    int z_contextual_idx = z + thread_index;
    std::vector<double> delta_output(outputs[z].size());
    for (int i = 0; i < outputs[z].size(); ++i) {
      // delta_output[i] = targets[i] - outputs[i];
      delta_output[i] = outputs[z][i] - targets[indexes[index_from + z]][i];
      // std::cout<<" C " << outputs[z][i] << " " << targets[indexes[index_from + z]][i]<<std::endl;
    }
    std::vector<double> d_z(delta_output.size());
    // std::vector<double> der_sigm_z_2 = derivative_activation(outputs, activation_function);
    /*for (int y = 0; y < delta_output.size(); ++y) {
      d_z[y] = delta_output[y];// * der_sigm_z_2[y];
      }*/
    d_z = delta_output;//derivative_activation(delta_output, "sigmoidal");
    for (int i = this->layers - 1; i >= 0; i--) {
      std::vector<double> der_sigm_z = derivative_activation(this->zetas[z_contextual_idx][i], this->activation_function);
      std::vector<double> prev_d_z = d_z;
      d_z = std::vector<double>(hidden_neurons[i], 0.0);
      for (int y = 0; y < this->weights[i + 1].size(); ++y) {
        for (int x = 0; x < this->weights[i + 1][y].size(); ++x) {
          d_z[x] += this->weights[i + 1][y][x] * prev_d_z[y] * der_sigm_z[x];
        }
      }
      for (int y = 0; y < prev_d_z.size(); ++y) {
        for (int x = 0; x < this->stored_activated_values[z_contextual_idx][i].size(); ++x) {
          this->weight_derivatives[i + 1][y][x] += prev_d_z[y] * this->stored_activated_values[z_contextual_idx][i][x];
        }
      }
    }
    //mu2.lock();
    for (int y = 0; y < d_z.size(); ++y) {
      for (int x = 0; x < X[indexes[index_from + z]].size(); ++x) {
        this->weight_derivatives[0][y][x] += d_z[y] * X[indexes[index_from + z]][x];
      }
    }
    //mu2.unlock();
    for (int i = 0; i < delta_output.size(); ++i) {
      // squared error
      error += std::abs(delta_output[i]);
    }
  }
  // std::cout<<"ERROR IS " << ( error / (double) X.size() )<<std::endl;
  return error;//pow(error / (double) (index_to - index_from), 2);
}

void MultiLayerPerceptron::update_weights(const int actual_batch_size) {

  for (int i = 0; i < this->weights.size(); ++i) {
    for (int y = 0; y < this->weights[i].size(); ++y) {
      for (int x = 0; x < this->weights[i][y].size(); ++x) {
        // Gradient descent
        // std::cout<< this->prev_velocities[i][y][x]<< " " << average_weight_derivative << " FF 1: " <<(average_weight_derivative - this->prev_velocities[i][y][x] / (double)  actual_batch_size)<<std::endl;

        double velocity_momentum = this->momentum * this->prev_velocities[i][y][x] + (1 - this->momentum) * learning_rate * this->weight_derivatives[i][y][x] / (double) actual_batch_size;
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

double MultiLayerPerceptron::forward_and_propagate(std::vector<std::vector<double> > &training_set, std::vector<std::vector<double> > &new_labels, std::vector<int> &indexes, int index_from, int index_to, int thread_index) {
  if (this->uses_gpu_backend()) {
    if (!this->training_backend->synchronize_model(this->export_flat_model())) {
      throw std::runtime_error(this->training_backend->last_error().empty() ? "GPU backend failed to synchronize model" : this->training_backend->last_error());
    }

    TrainingBatch batch = make_training_batch(training_set, indexes, index_from, index_to, this->size_outputs, &new_labels);
    BatchTrainingResult result;
    if (!this->training_backend->train_batch(batch, &result)) {
      throw std::runtime_error(this->training_backend->last_error().empty() ? "GPU backend failed during training batch" : this->training_backend->last_error());
    }
    if (static_cast<int>(result.layer_zetas.size()) != this->layers + 1 || static_cast<int>(result.layer_activations.size()) != this->layers + 1) {
      throw std::runtime_error("GPU backend returned incomplete layer activations");
    }
    if (static_cast<int>(result.layer_weight_gradients.size()) != this->layers + 1) {
      throw std::runtime_error("GPU backend returned incomplete weight gradients");
    }

    const int batch_size = index_to - index_from;
    for (int sample_index = 0; sample_index < batch_size; ++sample_index) {
      const int contextual_index = sample_index + thread_index;
      for (int layer_index = 0; layer_index < this->layers + 1; ++layer_index) {
        const LayerBatchValues &zetas = result.layer_zetas[layer_index];
        const LayerBatchValues &activations = result.layer_activations[layer_index];
        this->zetas[contextual_index][layer_index] = std::vector<double>(
          zetas.values.begin() + sample_index * zetas.value_size,
          zetas.values.begin() + (sample_index + 1) * zetas.value_size
        );
        this->stored_activated_values[contextual_index][layer_index] = std::vector<double>(
          activations.values.begin() + sample_index * activations.value_size,
          activations.values.begin() + (sample_index + 1) * activations.value_size
        );
      }
    }

    for (int layer_index = 0; layer_index < this->layers + 1; ++layer_index) {
      const FlatLayerWeights &gradient_layer = result.layer_weight_gradients[layer_index];
      int gradient_offset = 0;
      for (int neuron_index = 0; neuron_index < gradient_layer.output_size; ++neuron_index) {
        for (int input_index = 0; input_index < gradient_layer.input_size; ++input_index) {
          this->weight_derivatives[layer_index][neuron_index][input_index] += gradient_layer.values[gradient_offset++];
        }
      }
    }

    return result.loss;
  }

  std::vector<std::vector<double> > outputs = this->forward_propagation(training_set, indexes, index_from, index_to, thread_index);
  double loss = this->back_propagation(training_set, new_labels, outputs, indexes, index_from, index_to, thread_index);
  return loss;
}

int MultiLayerPerceptron::func(std::vector<int> &indexes) { return 1 * 2 * 3 * this->learning_rate; }

void MultiLayerPerceptron::initialize_for_input_size(int inputs_size) {
  if (first_time_training) {
    labels_unique_size = this->size_outputs;
    this->initialize(inputs_size);
    first_time_training = false;
  }
}

double MultiLayerPerceptron::train_single(std::vector<double> item, double label) {
  initialize_for_input_size(item.size());
  std::vector<std::vector<double> > training_set(1);
  std::vector<std::vector<double> > transformed_labels(1);
  std::vector<int> indexes(1);
  training_set[0] = item;
  transformed_labels[0] = transform_label(label);
  indexes[0] = 0;
  auto outputs = this->forward_propagation(training_set, indexes, 0, 1, 0);
  double loss = this->back_propagation(training_set, transformed_labels, outputs, indexes, 0, 1, 0);
  update_weights(1);
  return loss;
}

void MultiLayerPerceptron::train(std::vector< std::vector<double> > training_set, std::vector<double> labels) {
  std::cout.precision(17);
  if (first_time_training) {
    std::cout<<std::endl<<"Training size: "<<training_set.size()<<" with n. "<<epochs<<" epochs"<<std::endl;
    int inputs_size = training_set[0].size(); // must be same length as labels
    labels_unique_size = std::set<double>( labels.begin(), labels.end() ).size();
    std::cout<<"Inputs size "<<inputs_size << " with unique labels "<<labels_unique_size<<std::endl;
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
  auto rng = std::default_random_engine {};

  std::vector<int> indexes(training_set.size());
  for (int i = 0; i < training_set.size(); ++i) {
    indexes[i] = i;
  }
  int batch_count = 0;
  for (int i = 0; i < epochs; ++i) {
    auto begin = Clock::now();
    auto last_progress_update = begin;
    std::cout << "Starting epoch " << (i + 1) << "/" << epochs << std::endl;
    double loss = 0.0;
    // shuffle indexes
    std::shuffle(indexes.begin(), indexes.end(), r);
    int size = training_set.size() / this->batch_size;
    const int progress_interval = std::max(1, size / 20);
    for (int ind = 0; ind < size; ++ind) {
      int actual_batch_size = 0;
      if (training_set.size() / ((ind + 1) * this->batch_size) == 0) {
        actual_batch_size = training_set.size() % ((ind + 1) * this->batch_size); 
      } {
        actual_batch_size = this->batch_size;
      }
      int index_from = this->batch_size * ind;
      int index_to = this->batch_size * ind + actual_batch_size;

      // std::vector<std::vector<double> > outputs = this->forward_propagation(training_set, indexes, index_from, index_to);
      // loss += this->back_propagation(training_set, new_labels, outputs, indexes, index_from, index_to);
      // auto forward_propagate_thread = std::async(&MultiLayerPerceptron::func, this, std::ref(indexes));//, training_set, new_labels, indexes, index_from, index_to);
      if (this->uses_gpu_backend()) {
        loss += this->forward_and_propagate(training_set, new_labels, indexes, index_from, index_to, 0);
      } else {
        int block_size = this->batch_size / this->threads_size;
        std::vector< std::future<double> > futures(this->threads_size);
        for (int thread_idx = 0; thread_idx < this->threads_size; ++thread_idx) {
          int block_index_from = index_from + thread_idx * block_size;
          int block_index_to = index_from + (thread_idx + 1) * block_size;
          futures[thread_idx] = std::async(&MultiLayerPerceptron::forward_and_propagate, this, std::ref(training_set), std::ref(new_labels), std::ref(indexes), block_index_from, block_index_to, thread_idx * block_size);
        }
        for (auto &thread : futures) {
          loss += thread.get();
        }
      }
      // auto forward_propagate_thread = std::async(&MultiLayerPerceptron::forward_propagation, this, training_set, indexes, index_from, index_to);
      // auto forward_propagate_thread = std::async(&func);
      // loss += forward_propagate_thread.get();
      update_weights(actual_batch_size);
      if ((ind + 1) % progress_interval == 0 || ind + 1 == size) {
        const auto progress_now = Clock::now();
        const double seconds_since_last_update =
          std::chrono::duration_cast<std::chrono::milliseconds>(progress_now - last_progress_update).count() / 1000.0;
        const double epoch_elapsed_seconds =
          std::chrono::duration_cast<std::chrono::milliseconds>(progress_now - begin).count() / 1000.0;
        std::cout
          << "Epoch " << (i + 1) << "/" << epochs
          << " progress: " << (ind + 1) << "/" << size
          << " batches"
          << " | +" << seconds_since_last_update << "s"
          << " | elapsed " << epoch_elapsed_seconds << "s"
          << std::endl;
        last_progress_update = progress_now;
      }
    }
    std::cout<<"Epoch "<<(i + 1)<<"/"<<epochs<<", with error "<<loss/(double) training_set.size()<<std::endl;
    auto end = Clock::now();
    auto exec_time = end - begin;
    std::cout<<"Time for single epoch is "<< std::chrono::duration_cast<std::chrono::milliseconds>(exec_time).count()/(double) 1000<<std::endl;
  }
}

double MultiLayerPerceptron::predict(std::vector<double> item) {
  auto outputs = predict_proba(item);
  if (outputs.size() == 1) {
    if (activation_function == "sigmoidal") {
      std::cout<<"Value predicted, before returning actual class: "<<this->zetas[0][this->layers][0]<<std::endl;
      if (outputs[0] > 0.5) {
        return 1;
      } else {
        return 0;
      }
    } else {
      return this->zetas[0][this->layers][0];
    }
  } else {
    int predicted_output = -1;
    double predicted_value = -1;
    for (int i = 0; i < outputs.size(); ++i) {
      if (predicted_value < outputs[i]) {
        predicted_value = outputs[i];
        predicted_output = i;
      }
    }
    return predicted_output;
  }
}

std::vector<double> MultiLayerPerceptron::predict_proba(std::vector<double> item) {
  initialize_for_input_size(item.size());
  std::vector<std::vector<double> > input_item(1);
  std::vector<int> indexes(1);

  input_item[0] = item;
  indexes[0] = 0;
  auto outputs = this->forward_propagation(input_item, indexes, 0, 1, 0);
  return outputs[0];
}

std::vector<int> MultiLayerPerceptron::get_network_shape() const {
  std::vector<int> shape;
  if (this->weights.empty()) {
    return shape;
  }
  shape.push_back(this->weights[0][0].size());
  for (const auto &layer : this->weights) {
    shape.push_back(layer.size());
  }
  return shape;
}

std::vector<std::vector<double> > MultiLayerPerceptron::get_layer_activations() const {
  std::vector<std::vector<double> > activations;
  if (this->stored_activated_values.empty()) {
    return activations;
  }
  activations = this->stored_activated_values[0];
  if (!this->weights.empty() && activations.size() < this->weights.size()) {
    activations.resize(this->weights.size());
  }
  return activations;
}

std::vector<std::vector<std::vector<double> > > MultiLayerPerceptron::get_weights() const {
  return this->weights;
}

FlatNetworkModel MultiLayerPerceptron::export_flat_model() const {
  FlatNetworkModel model;
  model.shape = get_network_shape();
  model.activation_function = this->activation_function;
  model.output_size = this->size_outputs;

  model.layers.reserve(this->weights.size());
  for (const auto &layer : this->weights) {
    FlatLayerWeights flat_layer;
    flat_layer.output_size = layer.size();
    flat_layer.input_size = flat_layer.output_size == 0 ? 0 : layer[0].size();
    flat_layer.values.reserve(flat_layer.input_size * flat_layer.output_size);
    for (const auto &neuron_weights : layer) {
      flat_layer.values.insert(flat_layer.values.end(), neuron_weights.begin(), neuron_weights.end());
    }
    model.layers.push_back(std::move(flat_layer));
  }

  return model;
}

bool MultiLayerPerceptron::set_execution_backend(const std::string &backend_name, std::string *error_message) {
  if (backend_name == "cpu") {
    this->execution_backend = "cpu";
    this->training_backend.reset();
    return true;
  }

  if (backend_name != "gpu") {
    if (error_message != nullptr) {
      *error_message = "Unknown backend: " + backend_name;
    }
    return false;
  }

  auto backend = create_default_gpu_training_backend();
  if (!backend->is_available()) {
    if (error_message != nullptr) {
      *error_message = backend->last_error().empty() ? "GPU backend is unavailable" : backend->last_error();
    }
    return false;
  }
  if (!this->weights.empty() && !backend->initialize(this->export_flat_model())) {
    if (error_message != nullptr) {
      *error_message = backend->last_error().empty() ? "Failed to initialize GPU backend model state" : backend->last_error();
    }
    return false;
  }

  this->training_backend = backend;
  this->execution_backend = backend->name();
  return true;
}

std::string MultiLayerPerceptron::get_execution_backend() const {
  return this->execution_backend;
}

bool MultiLayerPerceptron::uses_gpu_backend() const {
  return this->execution_backend == "gpu" && this->training_backend != nullptr;
}
