#ifndef CSV_UTIL_H
#define CSV_UTIL_H

#include <eigen3/Eigen/Dense>
#include <vector>
#include <fstream>

#include <iostream>

// typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXi;

Eigen::MatrixXd read_csv(const std::string& path) {
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
	return Eigen::Map<Eigen::MatrixXd>(values.data(), rows, values.size()/rows);
}

template<typename M>
void write_csv(M& data, const std::string& path) { // work in progress
	std::ofstream outdata;
	outdata.open(path);
	return;
}

#endif

