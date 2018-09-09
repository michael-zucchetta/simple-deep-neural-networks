#ifndef MULTI_LAYER_PERCEPTRON_HPP 
#define MULTI_LAYER_PERCEPTRON_HPP 

#include <vector>
#include <string>

enum ActivationUnits { Relu, Sigmoid, Tanh, Linear };

class MultiLayerPerceptron {
  private:
    int layers;
    int epochs;
    int batch_size;
    int threads_size;
    int size_outputs;
    int labels_unique_size;
    float learning_rate;
    float  momentum;
    bool first_time_training;
    std::string activation_function;
    std::vector<std::vector<std::vector<double> > > weights;
    std::vector<std::vector<std::vector<double> > > weight_derivatives;
    std::vector<std::vector<std::vector<double> > > prev_velocities;
    std::vector<std::vector<std::vector<double> > > stored_activated_values;
    std::vector<std::vector<std::vector<double> > > zetas;
    int *hidden_neurons;
    double get_random();
    void initialize(int inputs_size);
    std::vector<double> transform_label(double label);
  public:
    MultiLayerPerceptron(int[], int, int, int, float, float, int, int, std::string);
    std::vector<std::vector<double> > forward_propagation(std::vector<std::vector<double> > &X, std::vector<int> &indexes, int index_from, int index_to, int thread_index);
    double forward_and_propagate(std::vector<std::vector<double> > &X, std::vector<std::vector<double> > &y, std::vector<int> &indexes, int index_from, int index_to, int thread_index);
    double back_propagation(std::vector<std::vector<double> > &X, std::vector<std::vector<double> > &y, std::vector<std::vector<double> > &outputs, std::vector<int> &indexes, int index_from, int index_to, int thread_index);
    int func(std::vector<int> &indexes);
    void update_weights(const int actual_batch_size);
    void train(std::vector< std::vector<double> > training_set, std::vector<double> labels);
    double predict(std::vector<double> item);
};
#endif
