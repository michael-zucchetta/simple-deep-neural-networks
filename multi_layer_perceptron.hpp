#ifndef MULTI_LAYER_PERCEPTRON_HPP 
#define MULTI_LAYER_PERCEPTRON_HPP 

#include <vector>
#include <string>

class MultiLayerPerceptron {
  private:
    unsigned int layers; 
    unsigned int epochs;
    int size_outputs;
    int labels_unique_size;
    float learning_rate;
    bool first_time_training;
    std::string activation_function;
    std::vector<std::vector<std::vector<double> > > weights;
    std::vector<std::vector<std::vector<double> > > weight_derivatives;
    std::vector<std::vector<double> > stored_activated_values;
    std::vector<std::vector<double> > zetas;
    unsigned int *hidden_neurons;
    double get_random();
    void initialize(unsigned int inputs_size);
    std::vector<double> transform_label(double label);
  public:
    MultiLayerPerceptron(unsigned int[], unsigned int, unsigned int, double, unsigned int, std::string);
    std::vector<double> forward_propagation(std::vector<double> X);
    double back_propagation(std::vector<double> X, std::vector<double> y, std::vector<double> outputs);
    void update_weights();
    void train(std::vector< std::vector<double> > training_set, std::vector<double> labels);
    double predict(std::vector<double> item);
};
#endif
