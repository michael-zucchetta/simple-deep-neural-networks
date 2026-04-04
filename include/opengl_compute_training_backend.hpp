#ifndef OPENGL_COMPUTE_TRAINING_BACKEND_HPP
#define OPENGL_COMPUTE_TRAINING_BACKEND_HPP

#include "gpu_training_backend.hpp"

#include <string>

class OpenGLComputeTrainingBackend : public GPUTrainingBackend {
  public:
    OpenGLComputeTrainingBackend();
    ~OpenGLComputeTrainingBackend() override;

    std::string name() const override;
    bool is_available() const override;
    const std::string &last_error() const override;
    bool initialize(const FlatNetworkModel &model) override;
    bool synchronize_model(const FlatNetworkModel &model) override;
    bool forward_batch(const TrainingBatch &batch, BatchTrainingResult *result) override;
    bool train_batch(const TrainingBatch &batch, BatchTrainingResult *result) override;

  private:
    bool initialize_context();
    bool initialize_pipeline();
    void destroy();
    void set_unavailable(const std::string &message);
    bool dispatch_forward_layer(
      const FlatLayerWeights &layer,
      const std::vector<double> &layer_inputs,
      int batch_size,
      std::vector<double> *zetas
    );
    bool dispatch_hidden_delta(
      const FlatLayerWeights &next_layer,
      const std::vector<double> &next_deltas,
      const std::vector<double> &current_zetas,
      int batch_size,
      int activation_mode,
      std::vector<double> *deltas
    );
    bool dispatch_weight_gradient(
      const std::vector<double> &previous_activations,
      int previous_size,
      const std::vector<double> &deltas,
      int output_size,
      int batch_size,
      std::vector<double> *gradients
    );

    std::string error_message;
    FlatNetworkModel current_model;
    void *window_handle = nullptr;
    unsigned int forward_program = 0;
    unsigned int hidden_delta_program = 0;
    unsigned int weight_gradient_program = 0;
    unsigned int weights_buffer = 0;
    unsigned int inputs_buffer = 0;
    unsigned int targets_buffer = 0;
    unsigned int outputs_buffer = 0;
    unsigned int gradients_buffer = 0;
    bool available = false;
};

#endif
