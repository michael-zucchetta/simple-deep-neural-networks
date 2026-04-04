#include "gpu_training_backend.hpp"

#include "opencl_training_backend.hpp"
#include "opengl_compute_training_backend.hpp"

#include <memory>
#include <string>

namespace {
class UnavailableGPUTrainingBackend : public GPUTrainingBackend {
  public:
    explicit UnavailableGPUTrainingBackend(std::string message)
      : error_message(std::move(message)) {}

    std::string name() const override {
      return "gpu";
    }

    bool is_available() const override {
      return false;
    }

    const std::string &last_error() const override {
      return error_message;
    }

    bool updates_weights_on_device() const override {
      return false;
    }

    bool initialize(const FlatNetworkModel &) override {
      return false;
    }

    bool synchronize_model(const FlatNetworkModel &) override {
      return false;
    }

    bool forward_batch(const TrainingBatch &, BatchTrainingResult *result) override {
      if (result != nullptr) {
        *result = BatchTrainingResult{};
      }
      return false;
    }

    bool train_batch(const TrainingBatch &, BatchTrainingResult *result) override {
      if (result != nullptr) {
        *result = BatchTrainingResult{};
      }
      return false;
    }

  private:
    std::string error_message;
};
}

std::shared_ptr<TrainingBackend> create_gpu_training_backend(const std::string &backend_name) {
  if (backend_name == "gpu" || backend_name == "opengl" || backend_name == "opengl-compute") {
#ifdef MLP_WITH_OPENGL_COMPUTE
    return std::make_shared<OpenGLComputeTrainingBackend>();
#else
    return std::make_shared<UnavailableGPUTrainingBackend>(
      "OpenGL compute backend is disabled in this build. Rebuild with OpenGL compute support using `make build-gpu`."
    );
#endif
  }

  if (backend_name == "opencl") {
    return std::make_shared<OpenCLTrainingBackend>();
  }

  return std::make_shared<UnavailableGPUTrainingBackend>(
    "Unknown GPU backend: " + backend_name
  );
}
