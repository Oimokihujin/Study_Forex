#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <vector>

#include "random_walker.hpp"

Random_Warker::Random_Warker(std::string fname){
    filename = fname;
    std::cout << "file::\t\t" << filename << std::endl;
}

Random_Warker::~Random_Warker(void){

}

int Random_Warker::readCSV(){
    std::fstream file;
    std::cout << filename << std::endl;
    file.open(filename, std::ios::in);
    
    if(!file.is_open()){
        std::cout << "Error in opening file." << std::endl;
        return EXIT_FAILURE;
    }

    std::string line;
    getline(file, line);

    // pass first line.
    
    while(getline(file, line)){
        std::string element;
        int array[] = {0, 1, 2, 3, 4};
        //std::cout << str << std::endl;
        for(auto& i : array){
            getline(line, element, ',');
            
        }
    }
    
    file.close();
        
    return EXIT_SUCCESS;    
}
