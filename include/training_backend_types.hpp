#ifndef TRAINING_BACKEND_TYPES_HPP
#define TRAINING_BACKEND_TYPES_HPP

#include <string>
#include <vector>

struct FlatLayerWeights {
  int input_size = 0;
  int output_size = 0;
  std::vector<double> values;
};

struct FlatNetworkModel {
  std::vector<int> shape;
  std::vector<FlatLayerWeights> layers;
  std::string activation_function;
  int output_size = 0;
  float learning_rate = 0.0f;
  float momentum = 0.0f;
};

struct TrainingBatch {
  int batch_size = 0;
  int input_size = 0;
  int output_size = 0;
  std::vector<double> inputs;
  std::vector<double> targets;
};

struct LayerBatchValues {
  int batch_size = 0;
  int value_size = 0;
  std::vector<double> values;
};

struct BatchTrainingResult {
  double loss = 0.0;
  std::vector<double> outputs;
  std::vector<double> gradients;
  std::vector<LayerBatchValues> layer_zetas;
  std::vector<LayerBatchValues> layer_activations;
  std::vector<FlatLayerWeights> layer_weight_gradients;
  std::vector<FlatLayerWeights> updated_weights;
  bool weights_updated_on_device = false;
};

#endif
