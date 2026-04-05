#include "opencl_training_backend.hpp"

#ifdef MLP_WITH_OPENCL

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

namespace {
const char *kOpenCLKernels = R"CLC(
__kernel void forward_layer(
  __global const float *weights,
  __global const float *inputs,
  __global float *outputs,
  const int input_size,
  const int output_size,
  const int batch_size
) {
  int output_index = get_global_id(0);
  int total_outputs = batch_size * output_size;
  if (output_index >= total_outputs) {
    return;
  }

  int sample_index = output_index / output_size;
  int neuron_index = output_index % output_size;
  float sum = 0.0f;
  for (int input_index = 0; input_index < input_size; ++input_index) {
    float weight = weights[neuron_index * input_size + input_index];
    float input_value = inputs[sample_index * input_size + input_index];
    sum += weight * input_value;
  }
  outputs[output_index] = sum;
}

inline float activation_derivative(float value, int activation_mode) {
  if (activation_mode == 1) {
    return value > 0.0f ? 1.0f : 0.0f;
  }
  if (activation_mode == 2) {
    float sigmoid = 1.0f / (1.0f + exp(-value));
    return sigmoid * (1.0f - sigmoid);
  }
  return 1.0f;
}

__kernel void hidden_delta(
  __global const float *weights,
  __global const float *next_deltas,
  __global const float *zetas,
  __global float *deltas,
  const int input_size,
  const int output_size,
  const int batch_size,
  const int activation_mode
) {
  int delta_index = get_global_id(0);
  int total_deltas = batch_size * input_size;
  if (delta_index >= total_deltas) {
    return;
  }

  int sample_index = delta_index / input_size;
  int neuron_index = delta_index % input_size;
  float propagated = 0.0f;
  for (int next_index = 0; next_index < output_size; ++next_index) {
    float weight = weights[next_index * input_size + neuron_index];
    float next_delta = next_deltas[sample_index * output_size + next_index];
    propagated += weight * next_delta;
  }
  deltas[delta_index] = propagated * activation_derivative(zetas[delta_index], activation_mode);
}

__kernel void weight_gradient(
  __global const float *activations,
  __global const float *deltas,
  __global float *gradients,
  const int input_size,
  const int output_size,
  const int batch_size
) {
  int gradient_index = get_global_id(0);
  int total_gradients = input_size * output_size;
  if (gradient_index >= total_gradients) {
    return;
  }

  int neuron_index = gradient_index / input_size;
  int input_index = gradient_index % input_size;
  float gradient_sum = 0.0f;
  for (int sample_index = 0; sample_index < batch_size; ++sample_index) {
    float activation = activations[sample_index * input_size + input_index];
    float delta = deltas[sample_index * output_size + neuron_index];
    gradient_sum += delta * activation;
  }
  gradients[gradient_index] = gradient_sum;
}

__kernel void apply_activation(
  __global const float *zetas,
  __global float *activations,
  const int value_size,
  const int batch_size,
  const int activation_mode
) {
  int value_index = get_global_id(0);
  int total_values = value_size * batch_size;
  if (value_index >= total_values) {
    return;
  }

  float value = zetas[value_index];
  if (activation_mode == 1) {
    activations[value_index] = value > 0.0f ? value : 0.0f;
    return;
  }
  if (activation_mode == 2) {
    activations[value_index] = 1.0f / (1.0f + exp(-value));
    return;
  }
  activations[value_index] = value;
}

__kernel void apply_softmax(
  __global const float *zetas,
  __global float *activations,
  const int value_size,
  const int batch_size
) {
  int sample_index = get_global_id(0);
  if (sample_index >= batch_size) {
    return;
  }

  float max_value = zetas[sample_index * value_size];
  for (int index = 1; index < value_size; ++index) {
    float value = zetas[sample_index * value_size + index];
    if (value > max_value) {
      max_value = value;
    }
  }

  float exp_sum = 0.0f;
  for (int index = 0; index < value_size; ++index) {
    float shifted = exp(zetas[sample_index * value_size + index] - max_value);
    activations[sample_index * value_size + index] = shifted;
    exp_sum += shifted;
  }

  if (exp_sum == 0.0f) {
    return;
  }
  for (int index = 0; index < value_size; ++index) {
    activations[sample_index * value_size + index] /= exp_sum;
  }
}

__kernel void update_weights(
  __global float *weights,
  __global const float *gradients,
  __global float *velocities,
  const int total_weights,
  const float learning_rate,
  const float momentum,
  const int batch_size
) {
  int weight_index = get_global_id(0);
  if (weight_index >= total_weights) {
    return;
  }

  float previous_velocity = velocities[weight_index];
  float average_gradient = gradients[weight_index] / (float) batch_size;
  float velocity_momentum = momentum * previous_velocity + (1.0f - momentum) * learning_rate * average_gradient;
  float velocity_nesterov = velocity_momentum + momentum * (velocity_momentum - previous_velocity);
  weights[weight_index] -= velocity_nesterov;
  velocities[weight_index] = velocity_nesterov;
}
)CLC";

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

std::string opencl_error_to_string(const cl_int error) {
  return "OpenCL error " + std::to_string(error);
}
}

OpenCLTrainingBackend::OpenCLTrainingBackend() {
  if (!initialize_context()) {
    return;
  }
  if (!initialize_pipeline()) {
    destroy();
    return;
  }
  available = true;
}

OpenCLTrainingBackend::~OpenCLTrainingBackend() {
  destroy();
}

std::string OpenCLTrainingBackend::name() const {
  return "opencl";
}

bool OpenCLTrainingBackend::is_available() const {
  return available;
}

const std::string &OpenCLTrainingBackend::last_error() const {
  return error_message;
}

bool OpenCLTrainingBackend::updates_weights_on_device() const {
  return true;
}

void OpenCLTrainingBackend::set_unavailable(const std::string &message) {
  available = false;
  error_message = message;
}

bool OpenCLTrainingBackend::initialize_context() {
  cl_uint platform_count = 0;
  cl_int status = clGetPlatformIDs(0, nullptr, &platform_count);
  if (status != CL_SUCCESS || platform_count == 0) {
    set_unavailable("Failed to enumerate OpenCL platforms");
    return false;
  }

  std::vector<cl_platform_id> platforms(platform_count);
  status = clGetPlatformIDs(platform_count, platforms.data(), nullptr);
  if (status != CL_SUCCESS) {
    set_unavailable("Failed to fetch OpenCL platforms: " + opencl_error_to_string(status));
    return false;
  }

  cl_device_id selected_device = nullptr;
  cl_platform_id selected_platform = nullptr;
  const cl_device_type preferred_types[] = {CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ACCELERATOR};
  for (cl_platform_id platform : platforms) {
    for (cl_device_type device_type : preferred_types) {
      cl_uint device_count = 0;
      status = clGetDeviceIDs(platform, device_type, 0, nullptr, &device_count);
      if (status != CL_SUCCESS || device_count == 0) {
        continue;
      }
      std::vector<cl_device_id> devices(device_count);
      status = clGetDeviceIDs(platform, device_type, device_count, devices.data(), nullptr);
      if (status == CL_SUCCESS && !devices.empty()) {
        selected_platform = platform;
        selected_device = devices.front();
        break;
      }
    }
    if (selected_device != nullptr) {
      break;
    }
  }

  if (selected_device == nullptr) {
    set_unavailable("No GPU/OpenCL accelerator device was found. CPU OpenCL devices are intentionally disabled for this backend.");
    return false;
  }

  cl_context context = clCreateContext(nullptr, 1, &selected_device, nullptr, nullptr, &status);
  if (status != CL_SUCCESS || context == nullptr) {
    set_unavailable("Failed to create OpenCL context: " + opencl_error_to_string(status));
    return false;
  }

  cl_command_queue queue = clCreateCommandQueue(context, selected_device, 0, &status);
  if (status != CL_SUCCESS || queue == nullptr) {
    clReleaseContext(context);
    set_unavailable("Failed to create OpenCL command queue: " + opencl_error_to_string(status));
    return false;
  }

  platform_id = selected_platform;
  device_id = selected_device;
  context_handle = context;
  queue_handle = queue;
  return true;
}

bool OpenCLTrainingBackend::initialize_pipeline() {
  cl_context context = static_cast<cl_context>(context_handle);
  cl_device_id device = static_cast<cl_device_id>(device_id);
  const char *source = kOpenCLKernels;
  const std::size_t source_length = std::strlen(source);
  cl_int status = CL_SUCCESS;
  cl_program program = clCreateProgramWithSource(context, 1, &source, &source_length, &status);
  if (status != CL_SUCCESS || program == nullptr) {
    set_unavailable("Failed to create OpenCL program: " + opencl_error_to_string(status));
    return false;
  }

  const char *build_options = "-cl-std=CL1.2 -Werror";
  char device_name[1024] = {0};
  char device_version[1024] = {0};
  char opencl_c_version[1024] = {0};
  clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
  clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(device_version), device_version, nullptr);
  clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, sizeof(opencl_c_version), opencl_c_version, nullptr);
  std::cerr << "OpenCL device: " << device_name << std::endl;
  std::cerr << "OpenCL device version: " << device_version << std::endl;
  std::cerr << "OpenCL C version: " << opencl_c_version << std::endl;
  status = clBuildProgram(program, 1, &device, build_options, nullptr, nullptr);
  if (status != CL_SUCCESS) {
    std::size_t log_size = 0;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    std::vector<char> build_log(log_size > 0 ? log_size : 1, '\0');
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, build_log.size(), build_log.data(), nullptr);
    std::size_t source_buffer_size = 0;
    clGetProgramInfo(program, CL_PROGRAM_SOURCE, 0, nullptr, &source_buffer_size);
    std::vector<char> source_buffer(source_buffer_size > 0 ? source_buffer_size : 1, '\0');
    clGetProgramInfo(program, CL_PROGRAM_SOURCE, source_buffer.size(), source_buffer.data(), nullptr);
    std::cerr << "OpenCL build failed, err=" << status << std::endl;
    std::cerr << "Build options: " << build_options << std::endl;
    std::cerr << "Build log:\n" << build_log.data() << std::endl;
    std::cerr << "Kernel source:\n" << source_buffer.data() << std::endl;
    clReleaseProgram(program);
    set_unavailable("Failed to build OpenCL program: " + std::string(build_log.data()));
    return false;
  }

  cl_kernel forward_kernel = clCreateKernel(program, "forward_layer", &status);
  if (status != CL_SUCCESS || forward_kernel == nullptr) {
    clReleaseProgram(program);
    set_unavailable("Failed to create OpenCL forward kernel: " + opencl_error_to_string(status));
    return false;
  }

  cl_kernel hidden_delta_kernel = clCreateKernel(program, "hidden_delta", &status);
  if (status != CL_SUCCESS || hidden_delta_kernel == nullptr) {
    clReleaseKernel(forward_kernel);
    clReleaseProgram(program);
    set_unavailable("Failed to create OpenCL hidden delta kernel: " + opencl_error_to_string(status));
    return false;
  }

  cl_kernel weight_gradient_kernel = clCreateKernel(program, "weight_gradient", &status);
  if (status != CL_SUCCESS || weight_gradient_kernel == nullptr) {
    clReleaseKernel(hidden_delta_kernel);
    clReleaseKernel(forward_kernel);
    clReleaseProgram(program);
    set_unavailable("Failed to create OpenCL weight gradient kernel: " + opencl_error_to_string(status));
    return false;
  }

  cl_kernel activation_kernel = clCreateKernel(program, "apply_activation", &status);
  if (status != CL_SUCCESS || activation_kernel == nullptr) {
    clReleaseKernel(weight_gradient_kernel);
    clReleaseKernel(hidden_delta_kernel);
    clReleaseKernel(forward_kernel);
    clReleaseProgram(program);
    set_unavailable("Failed to create OpenCL activation kernel: " + opencl_error_to_string(status));
    return false;
  }

  cl_kernel softmax_kernel = clCreateKernel(program, "apply_softmax", &status);
  if (status != CL_SUCCESS || softmax_kernel == nullptr) {
    clReleaseKernel(activation_kernel);
    clReleaseKernel(weight_gradient_kernel);
    clReleaseKernel(hidden_delta_kernel);
    clReleaseKernel(forward_kernel);
    clReleaseProgram(program);
    set_unavailable("Failed to create OpenCL softmax kernel: " + opencl_error_to_string(status));
    return false;
  }

  cl_kernel weight_update_kernel = clCreateKernel(program, "update_weights", &status);
  if (status != CL_SUCCESS || weight_update_kernel == nullptr) {
    clReleaseKernel(softmax_kernel);
    clReleaseKernel(activation_kernel);
    clReleaseKernel(weight_gradient_kernel);
    clReleaseKernel(hidden_delta_kernel);
    clReleaseKernel(forward_kernel);
    clReleaseProgram(program);
    set_unavailable("Failed to create OpenCL weight update kernel: " + opencl_error_to_string(status));
    return false;
  }

  program_handle = program;
  forward_kernel_handle = forward_kernel;
  hidden_delta_kernel_handle = hidden_delta_kernel;
  weight_gradient_kernel_handle = weight_gradient_kernel;
  activation_kernel_handle = activation_kernel;
  softmax_kernel_handle = softmax_kernel;
  weight_update_kernel_handle = weight_update_kernel;
  return true;
}

void OpenCLTrainingBackend::destroy() {
  release_layer_buffers();
  if (gradient_buffer_handle != nullptr) {
    clReleaseMemObject(static_cast<cl_mem>(gradient_buffer_handle));
    gradient_buffer_handle = nullptr;
  }
  if (secondary_output_buffer_handle != nullptr) {
    clReleaseMemObject(static_cast<cl_mem>(secondary_output_buffer_handle));
    secondary_output_buffer_handle = nullptr;
  }
  if (output_buffer_handle != nullptr) {
    clReleaseMemObject(static_cast<cl_mem>(output_buffer_handle));
    output_buffer_handle = nullptr;
  }
  if (secondary_input_buffer_handle != nullptr) {
    clReleaseMemObject(static_cast<cl_mem>(secondary_input_buffer_handle));
    secondary_input_buffer_handle = nullptr;
  }
  if (input_buffer_handle != nullptr) {
    clReleaseMemObject(static_cast<cl_mem>(input_buffer_handle));
    input_buffer_handle = nullptr;
  }
  input_buffer_capacity = 0;
  secondary_input_buffer_capacity = 0;
  output_buffer_capacity = 0;
  secondary_output_buffer_capacity = 0;
  gradient_buffer_capacity = 0;
  if (weight_update_kernel_handle != nullptr) {
    clReleaseKernel(static_cast<cl_kernel>(weight_update_kernel_handle));
    weight_update_kernel_handle = nullptr;
  }
  if (softmax_kernel_handle != nullptr) {
    clReleaseKernel(static_cast<cl_kernel>(softmax_kernel_handle));
    softmax_kernel_handle = nullptr;
  }
  if (activation_kernel_handle != nullptr) {
    clReleaseKernel(static_cast<cl_kernel>(activation_kernel_handle));
    activation_kernel_handle = nullptr;
  }
  if (weight_gradient_kernel_handle != nullptr) {
    clReleaseKernel(static_cast<cl_kernel>(weight_gradient_kernel_handle));
    weight_gradient_kernel_handle = nullptr;
  }
  if (hidden_delta_kernel_handle != nullptr) {
    clReleaseKernel(static_cast<cl_kernel>(hidden_delta_kernel_handle));
    hidden_delta_kernel_handle = nullptr;
  }
  if (forward_kernel_handle != nullptr) {
    clReleaseKernel(static_cast<cl_kernel>(forward_kernel_handle));
    forward_kernel_handle = nullptr;
  }
  if (program_handle != nullptr) {
    clReleaseProgram(static_cast<cl_program>(program_handle));
    program_handle = nullptr;
  }
  if (queue_handle != nullptr) {
    clReleaseCommandQueue(static_cast<cl_command_queue>(queue_handle));
    queue_handle = nullptr;
  }
  if (context_handle != nullptr) {
    clReleaseContext(static_cast<cl_context>(context_handle));
    context_handle = nullptr;
  }
  platform_id = nullptr;
  device_id = nullptr;
  available = false;
}

bool OpenCLTrainingBackend::initialize(const FlatNetworkModel &model) {
  current_model = model;
  if (!available) {
    return false;
  }
  if (current_model.layers.empty()) {
    set_unavailable("OpenCL backend initialization requires a non-empty network model");
    return false;
  }
  velocity_layers.clear();
  velocity_layers.reserve(current_model.layers.size());
  for (const FlatLayerWeights &layer : current_model.layers) {
    FlatLayerWeights velocity_layer;
    velocity_layer.input_size = layer.input_size;
    velocity_layer.output_size = layer.output_size;
    velocity_layer.values.assign(layer.values.size(), 0.0);
    velocity_layers.push_back(velocity_layer);
  }
  return initialize_layer_buffers();
}

bool OpenCLTrainingBackend::synchronize_model(const FlatNetworkModel &model) {
  current_model = model;
  if (velocity_layers.size() != current_model.layers.size()) {
    return initialize(model);
  }
  if (!initialize_layer_buffers()) {
    return false;
  }
  return available;
}

bool OpenCLTrainingBackend::ensure_buffer_capacity(void **buffer_handle, std::size_t *capacity, std::size_t element_count, unsigned long flags) {
  cl_context context = static_cast<cl_context>(context_handle);
  if (*buffer_handle != nullptr && *capacity >= element_count) {
    return true;
  }
  if (*buffer_handle != nullptr) {
    clReleaseMemObject(static_cast<cl_mem>(*buffer_handle));
    *buffer_handle = nullptr;
    *capacity = 0;
  }
  cl_int status = CL_SUCCESS;
  cl_mem buffer = clCreateBuffer(context, static_cast<cl_mem_flags>(flags), element_count * sizeof(float), nullptr, &status);
  if (status != CL_SUCCESS || buffer == nullptr) {
    set_unavailable("OpenCL buffer allocation failed: " + opencl_error_to_string(status));
    return false;
  }
  *buffer_handle = buffer;
  *capacity = element_count;
  return true;
}

bool OpenCLTrainingBackend::write_buffer(void *buffer_handle, const std::vector<float> &values) {
  cl_command_queue queue = static_cast<cl_command_queue>(queue_handle);
  const cl_int status = clEnqueueWriteBuffer(
    queue,
    static_cast<cl_mem>(buffer_handle),
    CL_TRUE,
    0,
    values.size() * sizeof(float),
    values.data(),
    0,
    nullptr,
    nullptr
  );
  if (status != CL_SUCCESS) {
    set_unavailable("OpenCL buffer write failed: " + opencl_error_to_string(status));
    return false;
  }
  return true;
}

bool OpenCLTrainingBackend::read_buffer(void *buffer_handle, std::size_t element_count, std::vector<float> *values) {
  cl_command_queue queue = static_cast<cl_command_queue>(queue_handle);
  values->assign(element_count, 0.0f);
  const cl_int status = clEnqueueReadBuffer(
    queue,
    static_cast<cl_mem>(buffer_handle),
    CL_TRUE,
    0,
    element_count * sizeof(float),
    values->data(),
    0,
    nullptr,
    nullptr
  );
  if (status != CL_SUCCESS) {
    set_unavailable("OpenCL buffer read failed: " + opencl_error_to_string(status));
    return false;
  }
  return true;
}

void OpenCLTrainingBackend::release_layer_buffers() {
  for (void *buffer_handle : layer_weight_buffers) {
    if (buffer_handle != nullptr) {
      clReleaseMemObject(static_cast<cl_mem>(buffer_handle));
    }
  }
  for (void *buffer_handle : layer_velocity_buffers) {
    if (buffer_handle != nullptr) {
      clReleaseMemObject(static_cast<cl_mem>(buffer_handle));
    }
  }
  layer_weight_buffers.clear();
  layer_velocity_buffers.clear();
}

bool OpenCLTrainingBackend::initialize_layer_buffers() {
  release_layer_buffers();
  layer_weight_buffers.resize(current_model.layers.size(), nullptr);
  layer_velocity_buffers.resize(velocity_layers.size(), nullptr);
  for (int layer_index = 0; layer_index < static_cast<int>(current_model.layers.size()); ++layer_index) {
    const FlatLayerWeights &layer = current_model.layers[layer_index];
    const FlatLayerWeights &velocity_layer = velocity_layers[layer_index];
    std::vector<float> weights(layer.values.begin(), layer.values.end());
    std::vector<float> velocities(velocity_layer.values.begin(), velocity_layer.values.end());
    cl_int status = CL_SUCCESS;
    layer_weight_buffers[layer_index] = clCreateBuffer(
      static_cast<cl_context>(context_handle),
      CL_MEM_READ_WRITE,
      weights.size() * sizeof(float),
      nullptr,
      &status
    );
    if (status != CL_SUCCESS || layer_weight_buffers[layer_index] == nullptr) {
      set_unavailable("OpenCL layer weight buffer allocation failed: " + opencl_error_to_string(status));
      return false;
    }
    layer_velocity_buffers[layer_index] = clCreateBuffer(
      static_cast<cl_context>(context_handle),
      CL_MEM_READ_WRITE,
      velocities.size() * sizeof(float),
      nullptr,
      &status
    );
    if (status != CL_SUCCESS || layer_velocity_buffers[layer_index] == nullptr) {
      set_unavailable("OpenCL layer velocity buffer allocation failed: " + opencl_error_to_string(status));
      return false;
    }
    if (!write_buffer(layer_weight_buffers[layer_index], weights) || !write_buffer(layer_velocity_buffers[layer_index], velocities)) {
      return false;
    }
  }
  return true;
}

bool OpenCLTrainingBackend::dispatch_forward_layer(
  const FlatLayerWeights &layer,
  const std::vector<double> &layer_inputs,
  int batch_size,
  std::vector<double> *zetas
) {
  cl_context context = static_cast<cl_context>(context_handle);
  cl_command_queue queue = static_cast<cl_command_queue>(queue_handle);
  cl_kernel kernel = static_cast<cl_kernel>(forward_kernel_handle);

  const std::size_t weights_size = layer.values.size();
  const std::size_t inputs_size = layer_inputs.size();
  const std::size_t outputs_size = static_cast<std::size_t>(batch_size * layer.output_size);
  std::vector<float> inputs(layer_inputs.begin(), layer_inputs.end());
  cl_int status = CL_SUCCESS;
  if (!ensure_buffer_capacity(&input_buffer_handle, &input_buffer_capacity, inputs_size, CL_MEM_READ_ONLY)) {
    return false;
  }
  if (!ensure_buffer_capacity(&output_buffer_handle, &output_buffer_capacity, outputs_size, CL_MEM_WRITE_ONLY)) {
    return false;
  }
  if (!write_buffer(input_buffer_handle, inputs)) {
    return false;
  }

  void *weights_buffer = layer_weight_buffers.empty() ? nullptr : layer_weight_buffers[&layer - &current_model.layers[0]];
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &weights_buffer);
  status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_buffer_handle);
  status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_buffer_handle);
  status |= clSetKernelArg(kernel, 3, sizeof(int), &layer.input_size);
  status |= clSetKernelArg(kernel, 4, sizeof(int), &layer.output_size);
  status |= clSetKernelArg(kernel, 5, sizeof(int), &batch_size);

  const size_t global_work_size = outputs_size;
  if (status == CL_SUCCESS) {
    status = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
  }
  if (status != CL_SUCCESS) {
    set_unavailable("OpenCL forward kernel execution failed: " + opencl_error_to_string(status));
    return false;
  }
  std::vector<float> outputs;
  if (!read_buffer(output_buffer_handle, outputs_size, &outputs)) {
    return false;
  }
  zetas->assign(outputs.begin(), outputs.end());
  return true;
}

bool OpenCLTrainingBackend::dispatch_hidden_delta(
  const FlatLayerWeights &next_layer,
  const std::vector<double> &next_deltas,
  const std::vector<double> &current_zetas,
  int batch_size,
  int activation_mode,
  std::vector<double> *deltas
) {
  cl_context context = static_cast<cl_context>(context_handle);
  cl_command_queue queue = static_cast<cl_command_queue>(queue_handle);
  cl_kernel kernel = static_cast<cl_kernel>(hidden_delta_kernel_handle);

  const std::size_t weights_size = next_layer.values.size();
  const std::size_t next_delta_size = next_deltas.size();
  const std::size_t current_zetas_size = current_zetas.size();
  const std::size_t outputs_size = static_cast<std::size_t>(batch_size * next_layer.input_size);
  std::vector<float> next_delta_values(next_deltas.begin(), next_deltas.end());
  std::vector<float> zeta_values(current_zetas.begin(), current_zetas.end());
  cl_int status = CL_SUCCESS;
  if (!ensure_buffer_capacity(&input_buffer_handle, &input_buffer_capacity, next_delta_size, CL_MEM_READ_ONLY)) {
    return false;
  }
  if (!ensure_buffer_capacity(&secondary_input_buffer_handle, &secondary_input_buffer_capacity, current_zetas_size, CL_MEM_READ_ONLY)) {
    return false;
  }
  if (!ensure_buffer_capacity(&output_buffer_handle, &output_buffer_capacity, outputs_size, CL_MEM_WRITE_ONLY)) {
    return false;
  }
  if (!write_buffer(input_buffer_handle, next_delta_values) || !write_buffer(secondary_input_buffer_handle, zeta_values)) {
    return false;
  }

  void *weights_buffer = layer_weight_buffers.empty() ? nullptr : layer_weight_buffers[&next_layer - &current_model.layers[0]];
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &weights_buffer);
  status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_buffer_handle);
  status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &secondary_input_buffer_handle);
  status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &output_buffer_handle);
  status |= clSetKernelArg(kernel, 4, sizeof(int), &next_layer.input_size);
  status |= clSetKernelArg(kernel, 5, sizeof(int), &next_layer.output_size);
  status |= clSetKernelArg(kernel, 6, sizeof(int), &batch_size);
  status |= clSetKernelArg(kernel, 7, sizeof(int), &activation_mode);

  const size_t global_work_size = outputs_size;
  if (status == CL_SUCCESS) {
    status = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
  }
  if (status != CL_SUCCESS) {
    set_unavailable("OpenCL hidden delta kernel execution failed: " + opencl_error_to_string(status));
    return false;
  }
  std::vector<float> output_deltas;
  if (!read_buffer(output_buffer_handle, outputs_size, &output_deltas)) {
    return false;
  }
  deltas->assign(output_deltas.begin(), output_deltas.end());
  return true;
}

bool OpenCLTrainingBackend::dispatch_weight_gradient(
  const std::vector<double> &previous_activations,
  int previous_size,
  const std::vector<double> &deltas,
  int output_size,
  int batch_size,
  std::vector<double> *gradients
) {
  cl_context context = static_cast<cl_context>(context_handle);
  cl_command_queue queue = static_cast<cl_command_queue>(queue_handle);
  cl_kernel kernel = static_cast<cl_kernel>(weight_gradient_kernel_handle);

  const std::size_t activations_size = previous_activations.size();
  const std::size_t delta_size = deltas.size();
  const std::size_t gradients_size = static_cast<std::size_t>(previous_size * output_size);
  std::vector<float> activation_values(previous_activations.begin(), previous_activations.end());
  std::vector<float> delta_values(deltas.begin(), deltas.end());
  cl_int status = CL_SUCCESS;
  if (!ensure_buffer_capacity(&input_buffer_handle, &input_buffer_capacity, activations_size, CL_MEM_READ_ONLY)) {
    return false;
  }
  if (!ensure_buffer_capacity(&secondary_input_buffer_handle, &secondary_input_buffer_capacity, delta_size, CL_MEM_READ_ONLY)) {
    return false;
  }
  if (!ensure_buffer_capacity(&gradient_buffer_handle, &gradient_buffer_capacity, gradients_size, CL_MEM_WRITE_ONLY)) {
    return false;
  }
  if (!write_buffer(input_buffer_handle, activation_values) || !write_buffer(secondary_input_buffer_handle, delta_values)) {
    return false;
  }

  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer_handle);
  status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &secondary_input_buffer_handle);
  status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &gradient_buffer_handle);
  status |= clSetKernelArg(kernel, 3, sizeof(int), &previous_size);
  status |= clSetKernelArg(kernel, 4, sizeof(int), &output_size);
  status |= clSetKernelArg(kernel, 5, sizeof(int), &batch_size);

  const size_t global_work_size = gradients_size;
  if (status == CL_SUCCESS) {
    status = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
  }
  if (status != CL_SUCCESS) {
    set_unavailable("OpenCL weight gradient kernel execution failed: " + opencl_error_to_string(status));
    return false;
  }
  std::vector<float> gradient_values;
  if (!read_buffer(gradient_buffer_handle, gradients_size, &gradient_values)) {
    return false;
  }
  gradients->assign(gradient_values.begin(), gradient_values.end());
  return true;
}

bool OpenCLTrainingBackend::dispatch_activation(
  const std::vector<double> &zetas,
  int value_size,
  int batch_size,
  int activation_mode,
  std::vector<double> *activations
) {
  cl_context context = static_cast<cl_context>(context_handle);
  cl_command_queue queue = static_cast<cl_command_queue>(queue_handle);
  cl_kernel kernel = static_cast<cl_kernel>(activation_kernel_handle);
  const std::size_t element_count = zetas.size();
  std::vector<float> zeta_values(zetas.begin(), zetas.end());
  cl_int status = CL_SUCCESS;
  if (!ensure_buffer_capacity(&input_buffer_handle, &input_buffer_capacity, element_count, CL_MEM_READ_ONLY)) {
    return false;
  }
  if (!ensure_buffer_capacity(&output_buffer_handle, &output_buffer_capacity, element_count, CL_MEM_WRITE_ONLY)) {
    return false;
  }
  if (!write_buffer(input_buffer_handle, zeta_values)) {
    return false;
  }

  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer_handle);
  status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer_handle);
  status |= clSetKernelArg(kernel, 2, sizeof(int), &value_size);
  status |= clSetKernelArg(kernel, 3, sizeof(int), &batch_size);
  status |= clSetKernelArg(kernel, 4, sizeof(int), &activation_mode);
  const size_t global_work_size = element_count;
  if (status == CL_SUCCESS) {
    status = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
  }
  if (status != CL_SUCCESS) {
    set_unavailable("OpenCL activation kernel execution failed: " + opencl_error_to_string(status));
    return false;
  }
  std::vector<float> activation_values;
  if (!read_buffer(output_buffer_handle, element_count, &activation_values)) {
    return false;
  }
  activations->assign(activation_values.begin(), activation_values.end());
  return true;
}

bool OpenCLTrainingBackend::dispatch_softmax(
  const std::vector<double> &zetas,
  int value_size,
  int batch_size,
  std::vector<double> *activations
) {
  cl_context context = static_cast<cl_context>(context_handle);
  cl_command_queue queue = static_cast<cl_command_queue>(queue_handle);
  cl_kernel kernel = static_cast<cl_kernel>(softmax_kernel_handle);
  const std::size_t element_count = zetas.size();
  std::vector<float> zeta_values(zetas.begin(), zetas.end());
  cl_int status = CL_SUCCESS;
  if (!ensure_buffer_capacity(&input_buffer_handle, &input_buffer_capacity, element_count, CL_MEM_READ_ONLY)) {
    return false;
  }
  if (!ensure_buffer_capacity(&output_buffer_handle, &output_buffer_capacity, element_count, CL_MEM_WRITE_ONLY)) {
    return false;
  }
  if (!write_buffer(input_buffer_handle, zeta_values)) {
    return false;
  }

  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer_handle);
  status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer_handle);
  status |= clSetKernelArg(kernel, 2, sizeof(int), &value_size);
  status |= clSetKernelArg(kernel, 3, sizeof(int), &batch_size);
  const size_t global_work_size = static_cast<size_t>(batch_size);
  if (status == CL_SUCCESS) {
    status = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
  }
  if (status != CL_SUCCESS) {
    set_unavailable("OpenCL softmax kernel execution failed: " + opencl_error_to_string(status));
    return false;
  }
  std::vector<float> activation_values;
  if (!read_buffer(output_buffer_handle, element_count, &activation_values)) {
    return false;
  }
  activations->assign(activation_values.begin(), activation_values.end());
  return true;
}

bool OpenCLTrainingBackend::dispatch_weight_update(
  int layer_index,
  const std::vector<double> &gradients,
  int input_size,
  int output_size,
  int batch_size,
  std::vector<double> *updated_weights
) {
  cl_context context = static_cast<cl_context>(context_handle);
  cl_command_queue queue = static_cast<cl_command_queue>(queue_handle);
  cl_kernel kernel = static_cast<cl_kernel>(weight_update_kernel_handle);
  FlatLayerWeights &weight_layer = current_model.layers[layer_index];
  FlatLayerWeights &velocity_layer = velocity_layers[layer_index];
  const std::size_t weight_count = weight_layer.values.size();
  std::vector<float> gradient_values(gradients.begin(), gradients.end());
  cl_int status = CL_SUCCESS;
  const float learning_rate = current_model.learning_rate;
  const float momentum = current_model.momentum;
  if (!ensure_buffer_capacity(&gradient_buffer_handle, &gradient_buffer_capacity, weight_count, CL_MEM_READ_ONLY)) {
    return false;
  }
  if (!write_buffer(gradient_buffer_handle, gradient_values)) {
    return false;
  }

  const int total_weights = static_cast<int>(weight_count);
  (void) input_size;
  (void) output_size;
  void *weights_buffer = layer_weight_buffers[layer_index];
  void *velocities_buffer = layer_velocity_buffers[layer_index];
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &weights_buffer);
  status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &gradient_buffer_handle);
  status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &velocities_buffer);
  status |= clSetKernelArg(kernel, 3, sizeof(int), &total_weights);
  status |= clSetKernelArg(kernel, 4, sizeof(float), &learning_rate);
  status |= clSetKernelArg(kernel, 5, sizeof(float), &momentum);
  status |= clSetKernelArg(kernel, 6, sizeof(int), &batch_size);
  const size_t global_work_size = weight_count;
  if (status == CL_SUCCESS) {
    status = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
  }
  if (status != CL_SUCCESS) {
    set_unavailable("OpenCL weight update kernel execution failed: " + opencl_error_to_string(status));
    return false;
  }
  std::vector<float> weights;
  std::vector<float> velocity_values;
  if (!read_buffer(weights_buffer, weight_count, &weights) || !read_buffer(velocities_buffer, weight_count, &velocity_values)) {
    return false;
  }
  weight_layer.values.assign(weights.begin(), weights.end());
  velocity_layer.values.assign(velocity_values.begin(), velocity_values.end());
  updated_weights->assign(weight_layer.values.begin(), weight_layer.values.end());
  return true;
}

bool OpenCLTrainingBackend::forward_batch(const TrainingBatch &batch, BatchTrainingResult *result) {
  if (result != nullptr) {
    *result = BatchTrainingResult{};
  }
  if (!available) {
    return false;
  }
  if (current_model.layers.empty()) {
    set_unavailable("OpenCL backend has not been initialized with model weights");
    return false;
  }
  if (batch.batch_size <= 0) {
    set_unavailable("OpenCL backend requires a non-empty batch");
    return false;
  }

  std::vector<double> layer_inputs = batch.inputs;
  const int layer_count = current_model.layers.size();
  if (result != nullptr) {
    result->layer_zetas.clear();
    result->layer_activations.clear();
  }

  for (int layer_index = 0; layer_index < layer_count; ++layer_index) {
    const FlatLayerWeights &layer = current_model.layers[layer_index];
    if (layer.input_size * batch.batch_size != static_cast<int>(layer_inputs.size())) {
      set_unavailable("OpenCL backend received mismatched layer input sizes");
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

    std::vector<double> next_inputs;
    if (layer_index + 1 == layer_count) {
      if (!dispatch_softmax(zetas, layer.output_size, batch.batch_size, &next_inputs)) {
        return false;
      }
    } else {
      if (!dispatch_activation(zetas, layer.output_size, batch.batch_size, activation_mode_from_name(current_model.activation_function), &next_inputs)) {
        return false;
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

bool OpenCLTrainingBackend::train_batch(const TrainingBatch &batch, BatchTrainingResult *result) {
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
  forward_result.updated_weights.resize(layer_count);

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

    std::vector<double> updated_weight_values;
    if (!dispatch_weight_update(layer_index, gradient_values, layer.input_size, layer.output_size, batch.batch_size, &updated_weight_values)) {
      if (result != nullptr) {
        *result = BatchTrainingResult{};
      }
      return false;
    }
    FlatLayerWeights updated_layer;
    updated_layer.input_size = layer.input_size;
    updated_layer.output_size = layer.output_size;
    updated_layer.values = updated_weight_values;
    forward_result.updated_weights[layer_index] = updated_layer;

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
    forward_result.weights_updated_on_device = true;
    *result = forward_result;
  }
  return true;
}

#else

OpenCLTrainingBackend::OpenCLTrainingBackend() {
  set_unavailable(
    "OpenCL backend is disabled in this build. Rebuild with `make build-opencl`."
  );
}

OpenCLTrainingBackend::~OpenCLTrainingBackend() {}

std::string OpenCLTrainingBackend::name() const {
  return "opencl";
}

bool OpenCLTrainingBackend::is_available() const {
  return available;
}

const std::string &OpenCLTrainingBackend::last_error() const {
  return error_message;
}

bool OpenCLTrainingBackend::updates_weights_on_device() const {
  return false;
}

bool OpenCLTrainingBackend::initialize(const FlatNetworkModel &) {
  return false;
}

bool OpenCLTrainingBackend::synchronize_model(const FlatNetworkModel &) {
  return false;
}

bool OpenCLTrainingBackend::forward_batch(const TrainingBatch &, BatchTrainingResult *result) {
  if (result != nullptr) {
    *result = BatchTrainingResult{};
  }
  return false;
}

bool OpenCLTrainingBackend::train_batch(const TrainingBatch &, BatchTrainingResult *result) {
  if (result != nullptr) {
    *result = BatchTrainingResult{};
  }
  return false;
}

bool OpenCLTrainingBackend::initialize_context() {
  return false;
}

bool OpenCLTrainingBackend::initialize_pipeline() {
  return false;
}

void OpenCLTrainingBackend::destroy() {}

void OpenCLTrainingBackend::set_unavailable(const std::string &message) {
  available = false;
  error_message = message;
}

bool OpenCLTrainingBackend::dispatch_forward_layer(
  const FlatLayerWeights &,
  const std::vector<double> &,
  int,
  std::vector<double> *
) {
  return false;
}

bool OpenCLTrainingBackend::dispatch_hidden_delta(
  const FlatLayerWeights &,
  const std::vector<double> &,
  const std::vector<double> &,
  int,
  int,
  std::vector<double> *
) {
  return false;
}

bool OpenCLTrainingBackend::dispatch_weight_gradient(
  const std::vector<double> &,
  int,
  const std::vector<double> &,
  int,
  int,
  std::vector<double> *
) {
  return false;
}

bool OpenCLTrainingBackend::dispatch_activation(
  const std::vector<double> &,
  int,
  int,
  int,
  std::vector<double> *
) {
  return false;
}

bool OpenCLTrainingBackend::dispatch_softmax(
  const std::vector<double> &,
  int,
  int,
  std::vector<double> *
) {
  return false;
}

bool OpenCLTrainingBackend::dispatch_weight_update(
  int,
  const std::vector<double> &,
  int,
  int,
  int,
  std::vector<double> *
) {
  return false;
}

#endif
