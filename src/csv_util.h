#ifndef CSV_UTIL_H
#define CSV_UTIL_H

#include <eigen3/Eigen/Dense>
#include <vector>
#include <fstream>

#include <iostream>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXdR;

MatrixXdR read_csv(const std::string& path) { // Output matrix needs to be RowMajor
	std::cerr << "Importing from csv: " << path << std::endl;
	std::ifstream indata;
	indata.open(path);
	std::string line;
	std::vector<double> values;
	uint rows = 0;
	while (std::getline(indata, line)) {
		std::stringstream lineStream(line);
		std::string cell;
		while (std::getline(lineStream, cell, ',')) {
			values.push_back(std::stod(cell));
		}
		++rows;
	}
	//int i = 0;
	//for (auto j: values) {
		//if (i%785==0)
			//if(j>0)
		//std::cout << j << ' ';
		//if (i>100*785)
			//break;
		//++i;
	//}
	return Eigen::Map<MatrixXdR>(values.data(), rows, values.size()/rows);
}

template<typename M>
void write_csv(M& data, const std::string& path) { // work in progress
	std::ofstream outdata;
	outdata.open(path);
	return;
}

#endif

