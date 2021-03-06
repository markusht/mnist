#include <iostream>
#include <cmath>

#include "csv_util.h"

std::tuple<MatrixXdR, MatrixXdR, MatrixXdR, MatrixXdR>
init_params() {
	std::cerr << "Initializing parameters" << std::endl;
	MatrixXdR W1 = MatrixXdR::Random(10,784)*0.5;
	MatrixXdR b1 = MatrixXdR::Random(10, 1)*0.5;
	MatrixXdR W2 = MatrixXdR::Random(10,10)*0.5;
	MatrixXdR b2 = MatrixXdR::Random(10, 1)*0.5;
	return {W1, b1, W2, b2};
}

std::tuple<MatrixXdR, MatrixXdR, MatrixXdR, MatrixXdR>
read_data() {
	const std::string trainPath = "../data/train.csv";
	const std::string testPath = "../data/test.csv";

	MatrixXdR data = read_csv(trainPath);
	//MatrixXdR data_test = read_csv(testPath);

	int m = data.rows(); // size of train data set
	int n = data.cols(); // amount of pixels + 1

	std::cerr << "Slicing raw data" << std::endl;
	MatrixXdR data_dev = data.block(0,0,1000,n);
	MatrixXdR data_train = data.block(1000,0, m - 1000, n);
	//data_test.transposeInPlace();
	data_dev.transposeInPlace();
	data_train.transposeInPlace();
	//MatrixXdR Y_dev = data_dev(0, Eigen::all);
	//MatrixXdR X_dev = data_dev(Eigen::seq(1,n-2), Eigen::all);
	MatrixXdR Y_dev = data_dev.block(0, 0, 1, data_dev.cols());
	MatrixXdR X_dev = data_dev.block(1, 0, data_dev.rows() - 1, data_dev.cols());
	//MatrixXdR Y_train = data_train(0, Eigen::all);
	//MatrixXdR X_train = data_train(Eigen::seq(1,n-1), Eigen::all);
	MatrixXdR Y_train = data_train.block(0, 0, 1, data_train.cols());
	MatrixXdR X_train = data_train.block(1, 0, data_train.rows() - 1, data_train.cols());

	return {Y_dev, X_dev, Y_train, X_train};
}

MatrixXdR ReLU(MatrixXdR Z) {
	return (Z.array() > 0).cast<double>() * Z.array();
}

MatrixXdR softmax(MatrixXdR Z) {
	//oj = exp (zj - m - log{sum_i{exp(zi-m)}})
	double m = Z.maxCoeff();
	MatrixXdR log_oj = (Z.array() - m).array() - log((Z.array()-m).array().exp().sum());
	MatrixXdR oj = (log_oj).array().exp();
	return oj;
}

std::tuple<MatrixXdR, MatrixXdR, MatrixXdR, MatrixXdR>
forward_prop(MatrixXdR W1, MatrixXdR b1, MatrixXdR W2, MatrixXdR b2, MatrixXdR X) {
	MatrixXdR Z1 = W1 * X + b1.replicate(1, X.cols());
	MatrixXdR A1 = ReLU(Z1);
	MatrixXdR Z2 = W2 * A1 + b2.replicate(1, A1.cols());
	MatrixXdR A2 = softmax(Z2);
	return {Z1, A1, Z2, A2};
}

MatrixXdR one_hot(MatrixXdR Y) {
	int m = Y.size();
	MatrixXdR one_hot_Y = MatrixXdR::Zero(m, 10); // because of 10 digits
	MatrixXdR identity = MatrixXdR::Identity(10, 10); // because of 10 digits
	for (int i=0; i<m; ++i) {
		one_hot_Y.row(i) = identity.row(Y(0, i));
	}
	one_hot_Y.transposeInPlace();
	return one_hot_Y;
}

MatrixXdR ReLU_deriv(MatrixXdR Z) {
	return (Z.array() > 0).cast<double>();
}

std::tuple<MatrixXdR, MatrixXdR, MatrixXdR, MatrixXdR>
backward_prop(MatrixXdR Z1, MatrixXdR A1, MatrixXdR Z2, MatrixXdR A2, MatrixXdR W1, MatrixXdR W2, MatrixXdR X, MatrixXdR Y) {
	double m = Y.size();
	MatrixXdR one_hot_Y = one_hot(Y);
	MatrixXdR dZ2 = A2 - one_hot_Y;
	MatrixXdR dW2 = 1 / m * dZ2 * A1.transpose();
	MatrixXdR db2 = 1 / m * dZ2.rowwise().sum();
	MatrixXdR dZ1 = (W2.transpose() * dZ2).array() * ReLU_deriv(Z1).array();
	MatrixXdR dW1 = 1 / m * dZ1 * X.transpose();
	MatrixXdR db1 = 1 / m * dZ1.rowwise().sum();
	return {dW1, db1, dW2, db2};
}

std::tuple<MatrixXdR, MatrixXdR, MatrixXdR, MatrixXdR>
update_params(MatrixXdR W1, MatrixXdR b1, MatrixXdR W2, MatrixXdR b2, MatrixXdR dW1, MatrixXdR db1, MatrixXdR dW2, MatrixXdR db2, double alpha) {
	W1 -= alpha * dW1;
	b1 -= alpha * db1;
	W2 -= alpha * dW2;
	b2 -= alpha * db2;
	return {W1, b1, W2, b2};
}

Eigen::VectorXi get_predictions(MatrixXdR A2) {
	int m = A2.cols();
	Eigen::VectorXi argmax0{m};
	for (int col = 0; col < m; ++col)
		A2.col(col).maxCoeff(&argmax0[col]);
	return argmax0;
}

double get_accuracy(Eigen::VectorXi predictions, MatrixXdR Y) {
	// std::cout << predictions << Y << std::endl;
	Eigen::VectorXd results = (predictions.array() == Y.transpose().array().cast<int>()).cast<double>();
	return results.sum() / (double)Y.size();
}

std::tuple<MatrixXdR, MatrixXdR, MatrixXdR, MatrixXdR>
gradient_descent(MatrixXdR X, MatrixXdR Y, double alpha, int iterations) {
	MatrixXdR W1, b1, W2, b2;
	W1.setZero();b1.setZero();W2.setZero();b2.setZero();
	std::tie(W1, b1, W2, b2) = init_params();
	for (int i=0; i<iterations; ++i) {
		MatrixXdR Z1, A1, Z2, A2;
		Z1.setZero();A1.setZero();Z2.setZero();A2.setZero();
		std::tie(Z1, A1, Z2, A2) = forward_prop(W1, b1, W2, b2, X);
		MatrixXdR dW1, db1, dW2, db2;
		dW1.setZero();db1.setZero();dW2.setZero();db2.setZero();
		std::tie(dW1, db1, dW2, db2) = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y);
		std::tie(W1, b1, W2, b2) = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha);
		if (i%1==0) {
			std::cerr << "Iteration: " << i << std::endl;
			Eigen::VectorXi predictions = get_predictions(A2);
			std::cerr << "Accuracy: " << get_accuracy(predictions, Y) << std::endl;
		}
	}
	return {W1, b1, W2, b2};
}

int main (int argc, char* argv[]) {
	MatrixXdR Y_dev, X_dev, Y_train, X_train;
	std::tie(Y_dev, X_dev, Y_train, X_train) = read_data();

	MatrixXdR W1, b1, W2, b2;
	std::tie(W1, b1, W2, b2) = gradient_descent(X_train, Y_train, 0.10, 500);

	//// For Testing
	//MatrixXdR test = MatrixXdR::Random(10,10)*1000;
	//int i = -50;
	//for (int m = 0; m<10; ++m) {
		//for (int n = 0; n<10; ++n) {
			//test(m,n) = i;
			//++i;
		//}
	//}
	//std::cout << "Unmodified array" << std::endl;
	//std::cout << test << std::endl;
	//std::cout << "Softmax" << std::endl;
	//std::cout << softmax(test) << std::endl;
	//std::cout << "ReLU" << std::endl;
	//std::cout << ReLU(test) << std::endl;

	return 0;
}
