#ifndef LETTERS_DATA_READER_HPP
#define LETTERS_DATA_READER_HPP
#include <tuple>
#include <fstream>

class LettersDataCSVReader {
  private:
    std::ifstream raw_data;
    double string_to_double( const std::string& s);
    auto parse_line(std::string line);
  public:
    std::tuple<std::vector< std::vector<double> >, std::vector<double>> read_letters_data();
};
#endif
