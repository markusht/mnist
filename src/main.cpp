#include <iostream>

#include "csv_util.h"

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
init_params() {
	Eigen::MatrixXd W1 = Eigen::MatrixXd::Random(10,784)*0.5;
	Eigen::MatrixXd b1 = Eigen::MatrixXd::Random(10,1)*0.5;
	Eigen::MatrixXd W2 = Eigen::MatrixXd::Random(10,10)*0.5;
	Eigen::MatrixXd b2 = Eigen::MatrixXd::Random(10,1)*0.5;
	Eigen::MatrixXd W3 = Eigen::MatrixXd::Random(10,10)*0.5;
	Eigen::MatrixXd b3 = Eigen::MatrixXd::Random(10,1)*0.5;
	return {W1, b1, W2, b2};
}

std::tuple<Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi>
read_data() {
	const std::string trainPath = "../data/train.csv";
	const std::string testPath = "../data/test.csv";

	Eigen::MatrixXi train = read_csv(trainPath);
	Eigen::MatrixXi test = read_csv(testPath);

	int m = train.rows(); // size of train data set
	int n = train.cols(); // amount of pixels + 1

	std::cerr << "Slicing raw data" << std::endl;
	Eigen::MatrixXi data_dev = test.transpose();
	Eigen::MatrixXi Y_dev = data_dev(0, Eigen::all);
	Eigen::MatrixXi X_dev = data_dev(Eigen::seq(1,n-2), Eigen::all);
	Eigen::MatrixXi data_train = train.transpose();
	Eigen::MatrixXi Y_train = data_train(0, Eigen::all);
	Eigen::MatrixXi X_train = data_train(Eigen::seq(1,n-1), Eigen::all);

	return {Y_dev, X_dev, Y_train, X_train};
}

int main (int argc, char* argv[]) {
	auto [Y_dev, X_dev, Y_train, X_train] = read_data();

	auto [W1, b1, W2, b2] = init_params();

	return 0;
}
