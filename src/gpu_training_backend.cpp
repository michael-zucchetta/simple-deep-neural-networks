#include "gpu_training_backend.hpp"

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

std::shared_ptr<TrainingBackend> create_default_gpu_training_backend() {
#ifdef MLP_WITH_OPENGL_COMPUTE
  return std::make_shared<OpenGLComputeTrainingBackend>();
#else
  return std::make_shared<UnavailableGPUTrainingBackend>(
    "GPU backend is disabled in this build. Rebuild with OpenGL compute support using `make build-gpu`."
  );
#endif
}
