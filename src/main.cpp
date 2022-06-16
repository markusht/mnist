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

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
read_data() {
	const std::string trainPath = "../data/train.csv";
	const std::string testPath = "../data/test.csv";

	Eigen::MatrixXd train = read_csv(trainPath);
	Eigen::MatrixXd test = read_csv(testPath);

	int m = train.rows(); // size of train data set
	int n = train.cols(); // amount of pixels + 1

	std::cerr << "Slicing raw data" << std::endl;
	Eigen::MatrixXd data_dev = test.transpose();
	Eigen::MatrixXd Y_dev = data_dev(0, Eigen::all);
	Eigen::MatrixXd X_dev = data_dev(Eigen::seq(1,n-2), Eigen::all);
	Eigen::MatrixXd data_train = train.transpose();
	Eigen::MatrixXd Y_train = data_train(0, Eigen::all);
	Eigen::MatrixXd X_train = data_train(Eigen::seq(1,n-1), Eigen::all);

	return {Y_dev, X_dev, Y_train, X_train};
}

Eigen::MatrixXd ReLU(Eigen::MatrixXd Z) {
	return Z.cwiseMax(0);
}

Eigen::MatrixXd softmax(Eigen::MatrixXd Z) {
	Eigen::MatrixXd m = Z.array().exp();
	double sum = m.sum();
	Eigen::MatrixXd n = m / sum;
	return n;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
forward_prop(Eigen::MatrixXd W1, Eigen::MatrixXd b1, Eigen::MatrixXd W2, Eigen::MatrixXd b2, Eigen::MatrixXd X) {
	Eigen::MatrixXd Z1 = W1*X + b1;
	Eigen::MatrixXd A1 = ReLU(Z1);
	Eigen::MatrixXd Z2 = W2*A1 + b2;
	Eigen::MatrixXd A2 = softmax(Z2);
	return {Z1, A1, Z2, A2};
}

Eigen::MatrixXd one_hot(Eigen::MatrixXd Y) {
	int m = Y.size();
	Eigen::MatrixXd one_hot_Y = Eigen::MatrixXd::Zero(m, 10); // 10 for number of digits
	Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(10, 10); // 10 for number of digits
	for (int i=0; i<m; ++i) {
		one_hot_Y.row(i) = identity.row(Y(0, i));
	}
	one_hot_Y = one_hot_Y.transpose();
	return one_hot_Y;
}

Eigen::MatrixXd ReLU_deriv(Eigen::MatrixXd Z) {
	return (Z.array() > 0).cast<double>();
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
back_prop(Eigen::MatrixXd Z1, Eigen::MatrixXd A1, Eigen::MatrixXd Z2, Eigen::MatrixXd A2, Eigen::MatrixXd W1, Eigen::MatrixXd W2, Eigen::MatrixXd X, Eigen::MatrixXd Y) {
	double m = Y.size();
	Eigen::MatrixXd one_hot_Y = one_hot(Y);
	Eigen::MatrixXd dZ2 = A2 - one_hot_Y;
	Eigen::MatrixXd dW2 = 1 / m * dZ2 * A1.transpose();
	Eigen::MatrixXd db2 = 1 / m * dZ2.rowwise().sum();
	Eigen::MatrixXd dZ1 = W2.transpose() * dZ2 * ReLU_deriv(Z1);
	Eigen::MatrixXd dW1 = 1 / m * dZ1 * X.transpose();
	Eigen::MatrixXd db1 = 1 / m * dZ1.rowwise().sum();
	return {dW1, db1, dW2, db2};
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
update_params(Eigen::MatrixXd W1, Eigen::MatrixXd b1, Eigen::MatrixXd W2, Eigen::MatrixXd b2, Eigen::MatrixXd dW1, Eigen::MatrixXd db1, Eigen::MatrixXd dW2, Eigen::MatrixXd db2, double alpha) {
	W1 = W1 - alpha * dW1;
	b1 = b1 - alpha * db1;
	W2 = W2 - alpha * dW2;
	b2 = b2 - alpha * db2;
	return {W1, b1, W2, b2};
}

int main (int argc, char* argv[]) {
	auto [Y_dev, X_dev, Y_train, X_train] = read_data();

	auto [W1, b1, W2, b2] = init_params();

	return 0;
}
