#ifndef GPU_TRAINING_BACKEND_HPP
#define GPU_TRAINING_BACKEND_HPP

#include "training_backend.hpp"

#include <memory>

class GPUTrainingBackend : public TrainingBackend {
  public:
    ~GPUTrainingBackend() override = default;
};

std::shared_ptr<TrainingBackend> create_default_gpu_training_backend();

#endif
