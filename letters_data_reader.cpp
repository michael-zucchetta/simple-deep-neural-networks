#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <tuple>
#include "letters_data_reader.hpp"

std::string DATASET_FILENAME = "./letter-recognition.data";
char SEPARATOR = ',';

double LettersDataCSVReader::string_to_double(const std::string& s) {
  std::istringstream i(s);
  double x;
  if (!(i >> x)) {
    return 0;
  }
  return x;
}

auto LettersDataCSVReader::parse_line(std::string line) {
  std::istringstream line_as_stream(line);
  std::vector<double> features;
  std::string label;
  std::getline(line_as_stream, label, SEPARATOR);
  // The classes start with 'A', so in this way it's possible to convert them to a char
  double label_as_short = label[0] - 'A';
  std::string doubleAsString;
  while(std::getline(line_as_stream, doubleAsString, SEPARATOR)) {
    features.push_back(string_to_double(doubleAsString));
  } 
  return std::make_tuple(features, label_as_short);
}
 
std::tuple<std::vector< std::vector<double> >, std::vector<double>> LettersDataCSVReader::read_letters_data() {
  std::string line;
  std::vector< std::vector<double> > dataset(0);
  std::vector<double> labels(0);
  raw_data.open(DATASET_FILENAME);
  if (raw_data.is_open()) {
    while (getline(raw_data, line)) {
      auto parsed_line = parse_line(line);
      double label = std::get<1>(parsed_line); 
      if (label > 26) {
        continue;
      }
      dataset.push_back(std::get<0>(parsed_line)); 
      labels.push_back(std::get<1>(parsed_line));
    } 
  }
  raw_data.close();
  return std::make_tuple(dataset, labels); 
}
/*
int main() {
  LettersDataCSVReader l = LettersDataCSVReader();
  auto dataset_and_labels = l.read_letters_data();
  std::vector<double> labels = std::get<1>(dataset_and_labels);
  for (int i = 0; i < labels.size(); ++i) {
    std::cout<<"\nOHI "<<labels[i]<<" "<<std::get<0>(dataset_and_labels)[0][0];
  }
}
*/
