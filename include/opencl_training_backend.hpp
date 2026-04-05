#ifndef OPENCL_TRAINING_BACKEND_HPP
#define OPENCL_TRAINING_BACKEND_HPP

#include "gpu_training_backend.hpp"

#include <string>

class OpenCLTrainingBackend : public GPUTrainingBackend {
  public:
    OpenCLTrainingBackend();
    ~OpenCLTrainingBackend() override;

    std::string name() const override;
    bool is_available() const override;
    const std::string &last_error() const override;
    bool updates_weights_on_device() const override;
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
    bool dispatch_activation(
      const std::vector<double> &zetas,
      int value_size,
      int batch_size,
      int activation_mode,
      std::vector<double> *activations
    );
    bool dispatch_softmax(
      const std::vector<double> &zetas,
      int value_size,
      int batch_size,
      std::vector<double> *activations
    );
    bool dispatch_weight_update(
      int layer_index,
      const std::vector<double> &gradients,
      int input_size,
      int output_size,
      int batch_size,
      std::vector<double> *updated_weights
    );
    bool ensure_buffer_capacity(void **buffer_handle, std::size_t *capacity, std::size_t element_count, unsigned long flags);
    bool write_buffer(void *buffer_handle, const std::vector<float> &values);
    bool read_buffer(void *buffer_handle, std::size_t element_count, std::vector<float> *values);
    bool initialize_layer_buffers();
    void release_layer_buffers();

    std::string error_message;
    FlatNetworkModel current_model;
    std::vector<FlatLayerWeights> velocity_layers;
    bool available = false;
    void *platform_id = nullptr;
    void *device_id = nullptr;
    void *context_handle = nullptr;
    void *queue_handle = nullptr;
    void *program_handle = nullptr;
    void *forward_kernel_handle = nullptr;
    void *hidden_delta_kernel_handle = nullptr;
    void *weight_gradient_kernel_handle = nullptr;
    void *activation_kernel_handle = nullptr;
    void *softmax_kernel_handle = nullptr;
    void *weight_update_kernel_handle = nullptr;
    std::vector<void *> layer_weight_buffers;
    std::vector<void *> layer_velocity_buffers;
    void *input_buffer_handle = nullptr;
    void *secondary_input_buffer_handle = nullptr;
    void *output_buffer_handle = nullptr;
    void *secondary_output_buffer_handle = nullptr;
    void *gradient_buffer_handle = nullptr;
    std::size_t input_buffer_capacity = 0;
    std::size_t secondary_input_buffer_capacity = 0;
    std::size_t output_buffer_capacity = 0;
    std::size_t secondary_output_buffer_capacity = 0;
    std::size_t gradient_buffer_capacity = 0;
};

#endif
