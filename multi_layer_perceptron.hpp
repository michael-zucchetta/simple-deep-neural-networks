#ifndef MULTI_LAYER_PERCEPTRON_HPP 
#define MULTI_LAYER_PERCEPTRON_HPP 

#include <vector>
#include <string>

class MultiLayerPerceptron {
  private:
    double hidden_neurons;
    unsigned int epochs;
    double size_outputs;
    double labels_unique_size;
    float learning_rate;
    bool first_time_training;
    std::string activation_function;
    std::vector<std::vector<double> > w1;
    std::vector<std::vector<double> > dw1;
    std::vector<std::vector<double> > w2;
    std::vector<std::vector<double> > dw2;
    std::vector<double> z1;
    std::vector<double> z2;
    std::vector<double> h;
    std::vector<double> o;
    double get_random();
    void randomize(unsigned int inputs_size);
    std::vector<double> transform_label(double label);
  public:
    MultiLayerPerceptron(double, double, int, double, std::string);
    void forward_propagation(std::vector<double> X);
    double back_propagation(std::vector<double> X, std::vector<double> y);
    void update_weights();
    void train(std::vector< std::vector<double> > training_set, std::vector<double> labels);
    double predict(std::vector<double> item);
};
#endif
