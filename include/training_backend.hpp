#ifndef TRAINING_BACKEND_HPP
#define TRAINING_BACKEND_HPP

#include "training_backend_types.hpp"

#include <string>

class TrainingBackend {
  public:
    virtual ~TrainingBackend() = default;

    virtual std::string name() const = 0;
    virtual bool is_available() const = 0;
    virtual const std::string &last_error() const = 0;
    virtual bool initialize(const FlatNetworkModel &model) = 0;
    virtual bool synchronize_model(const FlatNetworkModel &model) = 0;
    virtual bool forward_batch(const TrainingBatch &batch, BatchTrainingResult *result) = 0;
    virtual bool train_batch(const TrainingBatch &batch, BatchTrainingResult *result) = 0;
};

#endif
