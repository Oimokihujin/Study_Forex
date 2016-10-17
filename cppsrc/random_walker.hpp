#ifndef RANDOM_WARKER_HPP
#define RANDOM_WARKER_HPP

#include <string>
#include <vector>

class Random_Warker{
private:
    std::string filename;
    std::vector<float> opens, closes, lows, highs;
public:
    Random_Warker(std::string filename);
    virtual ~Random_Warker();
    int readCSV();

};
#endif
