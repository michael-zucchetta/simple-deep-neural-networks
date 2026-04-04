#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>
#include <emscripten/emscripten.h>
#include "multi_layer_perceptron.hpp"

namespace {
struct DemoSample {
  std::vector<double> input;
  double label;
};

class WasmDemo {
  public:
    WasmDemo() {
      xor_samples = {
        {{0.0, 0.0}, 0.0},
        {{0.0, 1.0}, 1.0},
        {{1.0, 0.0}, 1.0},
        {{1.0, 1.0}, 0.0}
      };
      hidden_layers = {6, 4};
      input_size = 2;
      output_size = 2;
      using_uploaded_dataset = false;
      reset();
    }

    void configure(const int *layers_ptr, int layers_count, int new_input_size, int new_output_size) {
      hidden_layers.clear();
      if (layers_ptr != nullptr && layers_count > 0) {
        hidden_layers.assign(layers_ptr, layers_ptr + layers_count);
      }
      input_size = std::max(1, new_input_size);
      output_size = std::max(1, new_output_size);
      using_uploaded_dataset = true;
      reset();
    }

    void clear_datasets() {
      training_samples.clear();
      validation_samples.clear();
      training_index = 0;
      validation_index = 0;
      current_training_sample_index = 0;
      completed_epochs = 0;
      reset();
    }

    void add_training_sample(const double *input_ptr, int length, double label) {
      training_samples.push_back(make_sample(input_ptr, length, label));
      if (training_samples.size() == 1) {
        reset();
      }
    }

    void add_validation_sample(const double *input_ptr, int length, double label) {
      validation_samples.push_back(make_sample(input_ptr, length, label));
      if (validation_samples.size() == 1) {
        select_validation_sample(0);
      }
    }

    void reset() {
      hidden_buffer = hidden_layers.empty() ? std::vector<int>{8} : hidden_layers;
      mlp = MultiLayerPerceptron(hidden_buffer.data(), 1, output_size, 1, 0.08f, 0.7f, hidden_buffer.size(), 1, "relu");
      mlp.initialize_for_input_size(input_size);
      training_index = 0;
      current_training_sample_index = 0;
      step_count = 0;
      completed_epochs = 0;
      last_loss = 0.0;
      current_label = 0.0;
      current_prediction = 0.0;
      current_probabilities.assign(output_size, output_size > 0 ? 1.0 / output_size : 0.0);

      const DemoSample *sample = current_training_sample();
      if (sample != nullptr) {
        last_input = sample->input;
        current_label = sample->label;
        current_probabilities = mlp.predict_proba(last_input);
        current_prediction = predicted_class_from_probabilities(current_probabilities);
      } else {
        last_input.assign(input_size, 0.0);
      }

      if (!validation_samples.empty()) {
        validation_index = std::min(validation_index, static_cast<int>(validation_samples.size()) - 1);
      } else {
        validation_index = 0;
      }
    }

    double step() {
      const DemoSample *sample = current_training_sample();
      if (sample == nullptr) {
        return last_loss;
      }

      current_training_sample_index = training_index;
      last_input = sample->input;
      current_label = sample->label;
      last_loss = mlp.train_single(sample->input, sample->label);
      current_probabilities = mlp.predict_proba(sample->input);
      current_prediction = predicted_class_from_probabilities(current_probabilities);

      ++training_index;
      if (training_index >= active_training_samples().size()) {
        training_index = 0;
        ++completed_epochs;
      }

      ++step_count;
      return last_loss;
    }

    int layer_count() const {
      return mlp.get_network_shape().size();
    }

    int layer_size(int layer_index) const {
      auto shape = mlp.get_network_shape();
      if (layer_index < 0 || layer_index >= static_cast<int>(shape.size())) {
        return 0;
      }
      return shape[layer_index];
    }

    double activation(int layer_index, int neuron_index) const {
      if (layer_index == 0) {
        if (neuron_index < 0 || neuron_index >= static_cast<int>(last_input.size())) {
          return 0.0;
        }
        return last_input[neuron_index];
      }

      auto activations = mlp.get_layer_activations();
      const int hidden_or_output_index = layer_index - 1;
      if (hidden_or_output_index < 0 || hidden_or_output_index >= static_cast<int>(activations.size())) {
        return 0.0;
      }
      if (neuron_index < 0 || neuron_index >= static_cast<int>(activations[hidden_or_output_index].size())) {
        return 0.0;
      }
      return activations[hidden_or_output_index][neuron_index];
    }

    double weight(int layer_index, int neuron_index, int input_index) const {
      auto weights = mlp.get_weights();
      if (layer_index <= 0 || layer_index > static_cast<int>(weights.size())) {
        return 0.0;
      }
      const auto &layer = weights[layer_index - 1];
      if (neuron_index < 0 || neuron_index >= static_cast<int>(layer.size())) {
        return 0.0;
      }
      if (input_index < 0 || input_index >= static_cast<int>(layer[neuron_index].size())) {
        return 0.0;
      }
      return layer[neuron_index][input_index];
    }

    double probability(int index) const {
      if (index < 0 || index >= static_cast<int>(current_probabilities.size())) {
        return 0.0;
      }
      return current_probabilities[index];
    }

    double target_label() const { return current_label; }
    double prediction() const { return current_prediction; }
    double loss() const { return last_loss; }
    int steps() const { return step_count; }
    int class_count() const { return output_size; }

    int training_sample_count() const {
      return active_training_samples().size();
    }

    int validation_sample_count() const {
      return validation_samples.size();
    }

    int epoch_count() const {
      return completed_epochs;
    }

    double training_progress() const {
      const auto &samples = active_training_samples();
      if (samples.empty()) {
        return 0.0;
      }
      return static_cast<double>(training_index) / static_cast<double>(samples.size());
    }

    int sample_index() const {
      return current_training_sample_index;
    }

    void select_validation_sample(int index) {
      if (validation_samples.empty()) {
        validation_index = 0;
        return;
      }
      validation_index = std::max(0, std::min(index, static_cast<int>(validation_samples.size()) - 1));
      const auto &sample = validation_samples[validation_index];
      current_probabilities = mlp.predict_proba(sample.input);
      current_prediction = predicted_class_from_probabilities(current_probabilities);
      current_label = sample.label;
    }

    int current_validation_sample_index() const {
      return validation_index;
    }

    double validation_pixel(int index) const {
      if (validation_samples.empty()) {
        return 0.0;
      }
      const auto &sample = validation_samples[validation_index];
      if (index < 0 || index >= static_cast<int>(sample.input.size())) {
        return 0.0;
      }
      return sample.input[index];
    }

    double validation_label() const {
      if (validation_samples.empty()) {
        return 0.0;
      }
      return validation_samples[validation_index].label;
    }

  private:
    std::vector<int> hidden_layers;
    std::vector<int> hidden_buffer;
    MultiLayerPerceptron mlp = MultiLayerPerceptron(nullptr, 1, 2, 1, 0.08f, 0.7f, 0, 1, "relu");
    std::vector<DemoSample> xor_samples;
    std::vector<DemoSample> training_samples;
    std::vector<DemoSample> validation_samples;
    std::vector<double> last_input;
    std::vector<double> current_probabilities;
    int input_size = 2;
    int output_size = 2;
    std::size_t training_index = 0;
    int current_training_sample_index = 0;
    int validation_index = 0;
    int step_count = 0;
    int completed_epochs = 0;
    double last_loss = 0.0;
    double current_label = 0.0;
    double current_prediction = 0.0;
    bool using_uploaded_dataset = false;

    const std::vector<DemoSample> &active_training_samples() const {
      if (using_uploaded_dataset) {
        return training_samples;
      }
      return xor_samples;
    }

    const DemoSample *current_training_sample() const {
      const auto &samples = active_training_samples();
      if (samples.empty()) {
        return nullptr;
      }
      return &samples[training_index % samples.size()];
    }

    DemoSample make_sample(const double *input_ptr, int length, double label) const {
      DemoSample sample;
      if (input_ptr != nullptr && length > 0) {
        sample.input.assign(input_ptr, input_ptr + length);
      } else {
        sample.input.assign(input_size, 0.0);
      }
      sample.label = label;
      return sample;
    }

    double predicted_class_from_probabilities(const std::vector<double> &probabilities) const {
      if (probabilities.empty()) {
        return 0.0;
      }

      int best_index = 0;
      double best_value = probabilities[0];
      for (int index = 1; index < static_cast<int>(probabilities.size()); ++index) {
        if (probabilities[index] > best_value) {
          best_value = probabilities[index];
          best_index = index;
        }
      }
      return static_cast<double>(best_index);
    }
};

WasmDemo &demo() {
  static WasmDemo instance;
  return instance;
}
}

extern "C" {
EMSCRIPTEN_KEEPALIVE void wasm_demo_reset() {
  demo().reset();
}

EMSCRIPTEN_KEEPALIVE double wasm_demo_step() {
  return demo().step();
}

EMSCRIPTEN_KEEPALIVE int wasm_demo_layer_count() {
  return demo().layer_count();
}

EMSCRIPTEN_KEEPALIVE int wasm_demo_layer_size(int layer_index) {
  return demo().layer_size(layer_index);
}

EMSCRIPTEN_KEEPALIVE double wasm_demo_get_activation(int layer_index, int neuron_index) {
  return demo().activation(layer_index, neuron_index);
}

EMSCRIPTEN_KEEPALIVE double wasm_demo_get_weight(int layer_index, int neuron_index, int input_index) {
  return demo().weight(layer_index, neuron_index, input_index);
}

EMSCRIPTEN_KEEPALIVE double wasm_demo_get_probability(int index) {
  return demo().probability(index);
}

EMSCRIPTEN_KEEPALIVE double wasm_demo_get_label() {
  return demo().target_label();
}

EMSCRIPTEN_KEEPALIVE double wasm_demo_get_prediction() {
  return demo().prediction();
}

EMSCRIPTEN_KEEPALIVE double wasm_demo_get_loss() {
  return demo().loss();
}

EMSCRIPTEN_KEEPALIVE int wasm_demo_get_step_count() {
  return demo().steps();
}

EMSCRIPTEN_KEEPALIVE void wasm_demo_configure(const int *layers_ptr, int layers_count, int input_size, int output_size) {
  demo().configure(layers_ptr, layers_count, input_size, output_size);
}

EMSCRIPTEN_KEEPALIVE void wasm_demo_clear_datasets() {
  demo().clear_datasets();
}

EMSCRIPTEN_KEEPALIVE void wasm_demo_add_training_sample(const double *input_ptr, int length, double label) {
  demo().add_training_sample(input_ptr, length, label);
}

EMSCRIPTEN_KEEPALIVE void wasm_demo_add_validation_sample(const double *input_ptr, int length, double label) {
  demo().add_validation_sample(input_ptr, length, label);
}

EMSCRIPTEN_KEEPALIVE void wasm_demo_select_validation_sample(int index) {
  demo().select_validation_sample(index);
}

EMSCRIPTEN_KEEPALIVE int wasm_demo_get_training_sample_count() {
  return demo().training_sample_count();
}

EMSCRIPTEN_KEEPALIVE int wasm_demo_get_validation_sample_count() {
  return demo().validation_sample_count();
}

EMSCRIPTEN_KEEPALIVE int wasm_demo_get_completed_epochs() {
  return demo().epoch_count();
}

EMSCRIPTEN_KEEPALIVE double wasm_demo_get_training_progress() {
  return demo().training_progress();
}

EMSCRIPTEN_KEEPALIVE int wasm_demo_get_current_sample_index() {
  return demo().sample_index();
}

EMSCRIPTEN_KEEPALIVE int wasm_demo_get_current_validation_index() {
  return demo().current_validation_sample_index();
}

EMSCRIPTEN_KEEPALIVE double wasm_demo_get_validation_pixel(int index) {
  return demo().validation_pixel(index);
}

EMSCRIPTEN_KEEPALIVE double wasm_demo_get_validation_label() {
  return demo().validation_label();
}

EMSCRIPTEN_KEEPALIVE int wasm_demo_get_class_count() {
  return demo().class_count();
}
}
