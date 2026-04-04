#include "opengl_compute_training_backend.hpp"

#ifdef MLP_WITH_OPENGL_COMPUTE

#include <algorithm>
#include <cmath>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

namespace {
const char *kForwardShaderSource = R"(
#version 430 core

layout(local_size_x = 64) in;

layout(std430, binding = 0) readonly buffer WeightsBuffer {
  float weights[];
};

layout(std430, binding = 1) readonly buffer InputsBuffer {
  float inputs[];
};

layout(std430, binding = 3) writeonly buffer OutputsBuffer {
  float outputs[];
};

uniform int uInputSize;
uniform int uOutputSize;
uniform int uBatchSize;

void main() {
  uint output_index = gl_GlobalInvocationID.x;
  uint total_outputs = uint(uBatchSize * uOutputSize);
  if (output_index >= total_outputs) {
    return;
  }

  uint sample_index = output_index / uint(uOutputSize);
  uint neuron_index = output_index % uint(uOutputSize);
  float sum = 0.0;
  for (int input_index = 0; input_index < uInputSize; ++input_index) {
    float weight = weights[neuron_index * uint(uInputSize) + uint(input_index)];
    float input_value = inputs[sample_index * uint(uInputSize) + uint(input_index)];
    sum += weight * input_value;
  }

  outputs[output_index] = sum;
}
)";

const char *kHiddenDeltaShaderSource = R"(
#version 430 core

layout(local_size_x = 64) in;

layout(std430, binding = 0) readonly buffer WeightsBuffer {
  float weights[];
};

layout(std430, binding = 1) readonly buffer InputsBuffer {
  float nextDeltas[];
};

layout(std430, binding = 2) readonly buffer TargetsBuffer {
  float zetas[];
};

layout(std430, binding = 3) writeonly buffer OutputsBuffer {
  float deltas[];
};

uniform int uInputSize;
uniform int uOutputSize;
uniform int uBatchSize;
uniform int uActivationMode;

float activation_derivative(float value, int activation_mode) {
  if (activation_mode == 1) {
    return value > 0.0 ? 1.0 : 0.0;
  }
  if (activation_mode == 2) {
    float sigmoid = 1.0 / (1.0 + exp(-value));
    return sigmoid * (1.0 - sigmoid);
  }
  return 1.0;
}

void main() {
  uint delta_index = gl_GlobalInvocationID.x;
  uint total_deltas = uint(uBatchSize * uInputSize);
  if (delta_index >= total_deltas) {
    return;
  }

  uint sample_index = delta_index / uint(uInputSize);
  uint neuron_index = delta_index % uint(uInputSize);
  float propagated = 0.0;
  for (int next_index = 0; next_index < uOutputSize; ++next_index) {
    float weight = weights[uint(next_index) * uint(uInputSize) + neuron_index];
    float next_delta = nextDeltas[sample_index * uint(uOutputSize) + uint(next_index)];
    propagated += weight * next_delta;
  }
  float zeta = zetas[delta_index];
  deltas[delta_index] = propagated * activation_derivative(zeta, uActivationMode);
}
)";

const char *kWeightGradientShaderSource = R"(
#version 430 core

layout(local_size_x = 64) in;

layout(std430, binding = 1) readonly buffer InputsBuffer {
  float activations[];
};

layout(std430, binding = 2) readonly buffer TargetsBuffer {
  float deltas[];
};

layout(std430, binding = 4) writeonly buffer GradientsBuffer {
  float gradients[];
};

uniform int uInputSize;
uniform int uOutputSize;
uniform int uBatchSize;

void main() {
  uint gradient_index = gl_GlobalInvocationID.x;
  uint total_gradients = uint(uInputSize * uOutputSize);
  if (gradient_index >= total_gradients) {
    return;
  }

  uint neuron_index = gradient_index / uint(uInputSize);
  uint input_index = gradient_index % uint(uInputSize);
  float gradient_sum = 0.0;
  for (int sample_index = 0; sample_index < uBatchSize; ++sample_index) {
    float activation = activations[sample_index * uint(uInputSize) + input_index];
    float delta = deltas[sample_index * uint(uOutputSize) + neuron_index];
    gradient_sum += delta * activation;
  }
  gradients[gradient_index] = gradient_sum;
}
)";

GLuint compile_compute_shader(const char *source, std::string *error_message) {
  GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);

  GLint success = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if (success == GL_TRUE) {
    return shader;
  }

  GLint log_length = 0;
  glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
  std::vector<char> info_log(log_length > 0 ? log_length : 1, '\0');
  glGetShaderInfoLog(shader, log_length, nullptr, info_log.data());
  if (error_message != nullptr) {
    *error_message = info_log.data();
  }
  glDeleteShader(shader);
  return 0;
}

std::vector<double> apply_activation_function(const std::vector<double> &values, const std::string &activation) {
  std::vector<double> activated_values(values.size(), 0.0);
  double sum = 0.0;
  for (int index = 0; index < static_cast<int>(values.size()); ++index) {
    double value = values[index];
    if (activation == "relu") {
      activated_values[index] = std::max(0.0, value);
    } else if (activation == "sigmoidal") {
      activated_values[index] = 1.0 / (1.0 + std::exp(-value));
    } else if (activation == "softmax") {
      activated_values[index] = std::exp(value);
      sum += activated_values[index];
    } else {
      activated_values[index] = value;
    }
  }
  if (activation == "softmax" && sum > 0.0) {
    for (double &value : activated_values) {
      value /= sum;
    }
  }
  return activated_values;
}

double average_absolute_error(const std::vector<double> &outputs, const std::vector<double> &targets, int batch_size, int output_size) {
  if (batch_size <= 0 || output_size <= 0) {
    return 0.0;
  }
  double accumulated = 0.0;
  const int total = batch_size * output_size;
  for (int index = 0; index < total; ++index) {
    accumulated += std::abs(outputs[index] - targets[index]);
  }
  return accumulated / static_cast<double>(batch_size);
}

int activation_mode_from_name(const std::string &activation) {
  if (activation == "relu") {
    return 1;
  }
  if (activation == "sigmoidal") {
    return 2;
  }
  return 0;
}
}

OpenGLComputeTrainingBackend::OpenGLComputeTrainingBackend() {
  if (!initialize_context()) {
    return;
  }
  if (!initialize_pipeline()) {
    destroy();
    return;
  }
  available = true;
}

OpenGLComputeTrainingBackend::~OpenGLComputeTrainingBackend() {
  destroy();
}

std::string OpenGLComputeTrainingBackend::name() const {
  return "gpu";
}

bool OpenGLComputeTrainingBackend::is_available() const {
  return available;
}

const std::string &OpenGLComputeTrainingBackend::last_error() const {
  return error_message;
}

void OpenGLComputeTrainingBackend::set_unavailable(const std::string &message) {
  available = false;
  error_message = message;
}

bool OpenGLComputeTrainingBackend::initialize_context() {
  if (glfwInit() != GLFW_TRUE) {
    set_unavailable("Failed to initialize GLFW for OpenGL compute backend");
    return false;
  }

  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow *window = glfwCreateWindow(1, 1, "mlp-gpu-compute", nullptr, nullptr);
  if (window == nullptr) {
    glfwTerminate();
    set_unavailable("Failed to create hidden OpenGL 4.3 context for compute backend");
    return false;
  }

  glfwMakeContextCurrent(window);
  glewExperimental = GL_TRUE;
  if (glewInit() != GLEW_OK) {
    glfwDestroyWindow(window);
    glfwTerminate();
    set_unavailable("Failed to initialize GLEW for compute backend");
    return false;
  }

  if (!GLEW_VERSION_4_3 && !GLEW_ARB_compute_shader) {
    glfwDestroyWindow(window);
    glfwTerminate();
    set_unavailable("OpenGL compute shaders require OpenGL 4.3 or ARB_compute_shader support");
    return false;
  }

  window_handle = window;
  return true;
}

bool OpenGLComputeTrainingBackend::initialize_pipeline() {
  std::string shader_error;
  GLuint forward_shader = compile_compute_shader(kForwardShaderSource, &shader_error);
  if (forward_shader == 0) {
    set_unavailable("Compute shader compilation failed: " + shader_error);
    return false;
  }

  forward_program = glCreateProgram();
  glAttachShader(forward_program, forward_shader);
  glLinkProgram(forward_program);
  glDeleteShader(forward_shader);

  GLint linked = 0;
  glGetProgramiv(forward_program, GL_LINK_STATUS, &linked);
  if (linked != GL_TRUE) {
    GLint log_length = 0;
    glGetProgramiv(forward_program, GL_INFO_LOG_LENGTH, &log_length);
    std::vector<char> info_log(log_length > 0 ? log_length : 1, '\0');
    glGetProgramInfoLog(forward_program, log_length, nullptr, info_log.data());
    set_unavailable("Compute shader link failed: " + std::string(info_log.data()));
    return false;
  }

  GLuint hidden_delta_shader = compile_compute_shader(kHiddenDeltaShaderSource, &shader_error);
  if (hidden_delta_shader == 0) {
    set_unavailable("Hidden delta shader compilation failed: " + shader_error);
    return false;
  }

  hidden_delta_program = glCreateProgram();
  glAttachShader(hidden_delta_program, hidden_delta_shader);
  glLinkProgram(hidden_delta_program);
  glDeleteShader(hidden_delta_shader);

  glGetProgramiv(hidden_delta_program, GL_LINK_STATUS, &linked);
  if (linked != GL_TRUE) {
    GLint log_length = 0;
    glGetProgramiv(hidden_delta_program, GL_INFO_LOG_LENGTH, &log_length);
    std::vector<char> info_log(log_length > 0 ? log_length : 1, '\0');
    glGetProgramInfoLog(hidden_delta_program, log_length, nullptr, info_log.data());
    set_unavailable("Hidden delta shader link failed: " + std::string(info_log.data()));
    return false;
  }

  GLuint weight_gradient_shader = compile_compute_shader(kWeightGradientShaderSource, &shader_error);
  if (weight_gradient_shader == 0) {
    set_unavailable("Weight gradient shader compilation failed: " + shader_error);
    return false;
  }

  weight_gradient_program = glCreateProgram();
  glAttachShader(weight_gradient_program, weight_gradient_shader);
  glLinkProgram(weight_gradient_program);
  glDeleteShader(weight_gradient_shader);

  glGetProgramiv(weight_gradient_program, GL_LINK_STATUS, &linked);
  if (linked != GL_TRUE) {
    GLint log_length = 0;
    glGetProgramiv(weight_gradient_program, GL_INFO_LOG_LENGTH, &log_length);
    std::vector<char> info_log(log_length > 0 ? log_length : 1, '\0');
    glGetProgramInfoLog(weight_gradient_program, log_length, nullptr, info_log.data());
    set_unavailable("Weight gradient shader link failed: " + std::string(info_log.data()));
    return false;
  }

  glGenBuffers(1, &weights_buffer);
  glGenBuffers(1, &inputs_buffer);
  glGenBuffers(1, &targets_buffer);
  glGenBuffers(1, &outputs_buffer);
  glGenBuffers(1, &gradients_buffer);

  return true;
}

void OpenGLComputeTrainingBackend::destroy() {
  if (forward_program != 0) {
    glDeleteProgram(forward_program);
    forward_program = 0;
  }
  if (hidden_delta_program != 0) {
    glDeleteProgram(hidden_delta_program);
    hidden_delta_program = 0;
  }
  if (weight_gradient_program != 0) {
    glDeleteProgram(weight_gradient_program);
    weight_gradient_program = 0;
  }
  if (weights_buffer != 0) {
    glDeleteBuffers(1, &weights_buffer);
    weights_buffer = 0;
  }
  if (inputs_buffer != 0) {
    glDeleteBuffers(1, &inputs_buffer);
    inputs_buffer = 0;
  }
  if (targets_buffer != 0) {
    glDeleteBuffers(1, &targets_buffer);
    targets_buffer = 0;
  }
  if (outputs_buffer != 0) {
    glDeleteBuffers(1, &outputs_buffer);
    outputs_buffer = 0;
  }
  if (gradients_buffer != 0) {
    glDeleteBuffers(1, &gradients_buffer);
    gradients_buffer = 0;
  }
  if (window_handle != nullptr) {
    GLFWwindow *window = static_cast<GLFWwindow *>(window_handle);
    glfwDestroyWindow(window);
    window_handle = nullptr;
    glfwTerminate();
  }
  available = false;
}

bool OpenGLComputeTrainingBackend::initialize(const FlatNetworkModel &model) {
  current_model = model;
  if (!available) {
    return false;
  }
  if (current_model.layers.empty()) {
    set_unavailable("GPU backend initialization requires a non-empty network model");
    return false;
  }
  return true;
}

bool OpenGLComputeTrainingBackend::synchronize_model(const FlatNetworkModel &model) {
  current_model = model;
  return available;
}

bool OpenGLComputeTrainingBackend::dispatch_forward_layer(
  const FlatLayerWeights &layer,
  const std::vector<double> &layer_inputs,
  int batch_size,
  std::vector<double> *zetas
) {
  const std::size_t outputs_size = static_cast<std::size_t>(batch_size * layer.output_size);
  std::vector<float> weights(layer.values.begin(), layer.values.end());
  std::vector<float> inputs(layer_inputs.begin(), layer_inputs.end());
  std::vector<float> outputs(outputs_size, 0.0f);

  glUseProgram(forward_program);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, weights_buffer);
  glBufferData(GL_SHADER_STORAGE_BUFFER, weights.size() * sizeof(float), weights.data(), GL_DYNAMIC_DRAW);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, weights_buffer);

  glBindBuffer(GL_SHADER_STORAGE_BUFFER, inputs_buffer);
  glBufferData(GL_SHADER_STORAGE_BUFFER, inputs.size() * sizeof(float), inputs.data(), GL_DYNAMIC_DRAW);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, inputs_buffer);

  glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputs_buffer);
  glBufferData(GL_SHADER_STORAGE_BUFFER, outputs.size() * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, outputs_buffer);

  glUniform1i(glGetUniformLocation(forward_program, "uInputSize"), layer.input_size);
  glUniform1i(glGetUniformLocation(forward_program, "uOutputSize"), layer.output_size);
  glUniform1i(glGetUniformLocation(forward_program, "uBatchSize"), batch_size);

  const GLuint work_groups = static_cast<GLuint>((outputs_size + 63) / 64);
  glDispatchCompute(work_groups, 1, 1);
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

  glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputs_buffer);
  glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, outputs.size() * sizeof(float), outputs.data());
  zetas->assign(outputs.begin(), outputs.end());
  return true;
}

bool OpenGLComputeTrainingBackend::dispatch_hidden_delta(
  const FlatLayerWeights &next_layer,
  const std::vector<double> &next_deltas,
  const std::vector<double> &current_zetas,
  int batch_size,
  int activation_mode,
  std::vector<double> *deltas
) {
  const std::size_t delta_count = static_cast<std::size_t>(batch_size * next_layer.input_size);
  std::vector<float> weights(next_layer.values.begin(), next_layer.values.end());
  std::vector<float> input_deltas(next_deltas.begin(), next_deltas.end());
  std::vector<float> zetas(current_zetas.begin(), current_zetas.end());
  std::vector<float> output_deltas(delta_count, 0.0f);

  glUseProgram(hidden_delta_program);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, weights_buffer);
  glBufferData(GL_SHADER_STORAGE_BUFFER, weights.size() * sizeof(float), weights.data(), GL_DYNAMIC_DRAW);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, weights_buffer);

  glBindBuffer(GL_SHADER_STORAGE_BUFFER, inputs_buffer);
  glBufferData(GL_SHADER_STORAGE_BUFFER, input_deltas.size() * sizeof(float), input_deltas.data(), GL_DYNAMIC_DRAW);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, inputs_buffer);

  glBindBuffer(GL_SHADER_STORAGE_BUFFER, targets_buffer);
  glBufferData(GL_SHADER_STORAGE_BUFFER, zetas.size() * sizeof(float), zetas.data(), GL_DYNAMIC_DRAW);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, targets_buffer);

  glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputs_buffer);
  glBufferData(GL_SHADER_STORAGE_BUFFER, output_deltas.size() * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, outputs_buffer);

  glUniform1i(glGetUniformLocation(hidden_delta_program, "uInputSize"), next_layer.input_size);
  glUniform1i(glGetUniformLocation(hidden_delta_program, "uOutputSize"), next_layer.output_size);
  glUniform1i(glGetUniformLocation(hidden_delta_program, "uBatchSize"), batch_size);
  glUniform1i(glGetUniformLocation(hidden_delta_program, "uActivationMode"), activation_mode);

  const GLuint work_groups = static_cast<GLuint>((delta_count + 63) / 64);
  glDispatchCompute(work_groups, 1, 1);
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

  glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputs_buffer);
  glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, output_deltas.size() * sizeof(float), output_deltas.data());
  deltas->assign(output_deltas.begin(), output_deltas.end());
  return true;
}

bool OpenGLComputeTrainingBackend::dispatch_weight_gradient(
  const std::vector<double> &previous_activations,
  int previous_size,
  const std::vector<double> &deltas,
  int output_size,
  int batch_size,
  std::vector<double> *gradients
) {
  const std::size_t gradient_count = static_cast<std::size_t>(previous_size * output_size);
  std::vector<float> activations(previous_activations.begin(), previous_activations.end());
  std::vector<float> delta_values(deltas.begin(), deltas.end());
  std::vector<float> gradient_values(gradient_count, 0.0f);

  glUseProgram(weight_gradient_program);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, inputs_buffer);
  glBufferData(GL_SHADER_STORAGE_BUFFER, activations.size() * sizeof(float), activations.data(), GL_DYNAMIC_DRAW);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, inputs_buffer);

  glBindBuffer(GL_SHADER_STORAGE_BUFFER, targets_buffer);
  glBufferData(GL_SHADER_STORAGE_BUFFER, delta_values.size() * sizeof(float), delta_values.data(), GL_DYNAMIC_DRAW);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, targets_buffer);

  glBindBuffer(GL_SHADER_STORAGE_BUFFER, gradients_buffer);
  glBufferData(GL_SHADER_STORAGE_BUFFER, gradient_values.size() * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, gradients_buffer);

  glUniform1i(glGetUniformLocation(weight_gradient_program, "uInputSize"), previous_size);
  glUniform1i(glGetUniformLocation(weight_gradient_program, "uOutputSize"), output_size);
  glUniform1i(glGetUniformLocation(weight_gradient_program, "uBatchSize"), batch_size);

  const GLuint work_groups = static_cast<GLuint>((gradient_count + 63) / 64);
  glDispatchCompute(work_groups, 1, 1);
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

  glBindBuffer(GL_SHADER_STORAGE_BUFFER, gradients_buffer);
  glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, gradient_values.size() * sizeof(float), gradient_values.data());
  gradients->assign(gradient_values.begin(), gradient_values.end());
  return true;
}

bool OpenGLComputeTrainingBackend::forward_batch(const TrainingBatch &batch, BatchTrainingResult *result) {
  if (result != nullptr) {
    *result = BatchTrainingResult{};
  }
  if (!available) {
    return false;
  }
  if (current_model.layers.empty()) {
    set_unavailable("GPU backend has not been initialized with model weights");
    return false;
  }

  if (batch.batch_size <= 0 || current_model.layers.empty()) {
    set_unavailable("OpenGL compute backend requires a non-empty model and batch");
    return false;
  }

  glfwMakeContextCurrent(static_cast<GLFWwindow *>(window_handle));
  glUseProgram(forward_program);

  std::vector<double> layer_inputs = batch.inputs;
  const int layer_count = current_model.layers.size();
  if (result != nullptr) {
    result->layer_zetas.clear();
    result->layer_activations.clear();
  }

  for (int layer_index = 0; layer_index < layer_count; ++layer_index) {
    const FlatLayerWeights &layer = current_model.layers[layer_index];
    if (layer.input_size * batch.batch_size != static_cast<int>(layer_inputs.size())) {
      set_unavailable("OpenGL compute backend received mismatched layer input sizes");
      return false;
    }

    std::vector<double> zetas;
    if (!dispatch_forward_layer(layer, layer_inputs, batch.batch_size, &zetas)) {
      return false;
    }
    if (result != nullptr) {
      LayerBatchValues layer_zetas;
      layer_zetas.batch_size = batch.batch_size;
      layer_zetas.value_size = layer.output_size;
      layer_zetas.values = zetas;
      result->layer_zetas.push_back(layer_zetas);
    }

    const std::size_t layer_output_count = static_cast<std::size_t>(batch.batch_size * layer.output_size);
    std::vector<double> next_inputs(layer_output_count, 0.0);
    for (int sample_index = 0; sample_index < batch.batch_size; ++sample_index) {
      std::vector<double> sample_values(layer.output_size, 0.0);
      for (int neuron_index = 0; neuron_index < layer.output_size; ++neuron_index) {
        sample_values[neuron_index] = zetas[sample_index * layer.output_size + neuron_index];
      }
      const std::string activation_name = layer_index + 1 == layer_count ? "softmax" : current_model.activation_function;
      std::vector<double> activated = apply_activation_function(sample_values, activation_name);
      for (int neuron_index = 0; neuron_index < layer.output_size; ++neuron_index) {
        next_inputs[sample_index * layer.output_size + neuron_index] = activated[neuron_index];
      }
    }

    if (result != nullptr) {
      LayerBatchValues layer_activations;
      layer_activations.batch_size = batch.batch_size;
      layer_activations.value_size = layer.output_size;
      layer_activations.values = next_inputs;
      result->layer_activations.push_back(layer_activations);
    }

    layer_inputs = next_inputs;
  }

  if (result != nullptr) {
    result->outputs = layer_inputs;
    result->loss = average_absolute_error(result->outputs, batch.targets, batch.batch_size, batch.output_size);
    result->gradients.resize(result->outputs.size(), 0.0);
    for (int index = 0; index < static_cast<int>(result->outputs.size()); ++index) {
      result->gradients[index] = result->outputs[index] - batch.targets[index];
    }
  }
  return true;
}

bool OpenGLComputeTrainingBackend::train_batch(const TrainingBatch &batch, BatchTrainingResult *result) {
  BatchTrainingResult forward_result;
  if (!forward_batch(batch, &forward_result)) {
    if (result != nullptr) {
      *result = BatchTrainingResult{};
    }
    return false;
  }

  const int layer_count = current_model.layers.size();
  std::vector<double> current_deltas = forward_result.gradients;
  forward_result.layer_weight_gradients.resize(layer_count);

  for (int layer_index = layer_count - 1; layer_index >= 0; --layer_index) {
    const FlatLayerWeights &layer = current_model.layers[layer_index];
    const std::vector<double> &previous_activations =
      layer_index == 0 ? batch.inputs : forward_result.layer_activations[layer_index - 1].values;

    std::vector<double> gradient_values;
    if (!dispatch_weight_gradient(previous_activations, layer.input_size, current_deltas, layer.output_size, batch.batch_size, &gradient_values)) {
      if (result != nullptr) {
        *result = BatchTrainingResult{};
      }
      return false;
    }

    FlatLayerWeights gradient_layer;
    gradient_layer.input_size = layer.input_size;
    gradient_layer.output_size = layer.output_size;
    gradient_layer.values = gradient_values;
    forward_result.layer_weight_gradients[layer_index] = gradient_layer;

    if (layer_index > 0) {
      std::vector<double> previous_deltas;
      if (!dispatch_hidden_delta(
            layer,
            current_deltas,
            forward_result.layer_zetas[layer_index - 1].values,
            batch.batch_size,
            activation_mode_from_name(current_model.activation_function),
            &previous_deltas
          )) {
        if (result != nullptr) {
          *result = BatchTrainingResult{};
        }
        return false;
      }
      current_deltas = previous_deltas;
    }
  }

  if (result != nullptr) {
    *result = forward_result;
  }
  return true;
}

#else

OpenGLComputeTrainingBackend::OpenGLComputeTrainingBackend() {
  set_unavailable(
    "OpenGL compute backend is disabled in this build. Rebuild with `make build-gpu`."
  );
}

OpenGLComputeTrainingBackend::~OpenGLComputeTrainingBackend() {}

std::string OpenGLComputeTrainingBackend::name() const {
  return "gpu";
}

bool OpenGLComputeTrainingBackend::is_available() const {
  return available;
}

const std::string &OpenGLComputeTrainingBackend::last_error() const {
  return error_message;
}

bool OpenGLComputeTrainingBackend::initialize(const FlatNetworkModel &) {
  return false;
}

bool OpenGLComputeTrainingBackend::synchronize_model(const FlatNetworkModel &) {
  return false;
}

bool OpenGLComputeTrainingBackend::forward_batch(const TrainingBatch &, BatchTrainingResult *result) {
  if (result != nullptr) {
    *result = BatchTrainingResult{};
  }
  return false;
}

bool OpenGLComputeTrainingBackend::train_batch(const TrainingBatch &, BatchTrainingResult *result) {
  if (result != nullptr) {
    *result = BatchTrainingResult{};
  }
  return false;
}

bool OpenGLComputeTrainingBackend::initialize_context() {
  return false;
}

bool OpenGLComputeTrainingBackend::initialize_pipeline() {
  return false;
}

void OpenGLComputeTrainingBackend::destroy() {}

void OpenGLComputeTrainingBackend::set_unavailable(const std::string &message) {
  available = false;
  error_message = message;
}

#endif
