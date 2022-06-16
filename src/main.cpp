#include <iostream>

#include "csv_util.h"

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
init_params() {
	std::cerr << "Initializing parameters" << std::endl;
	Eigen::MatrixXd W1 = Eigen::MatrixXd::Random(10,784)*0.5;
	Eigen::MatrixXd b1 = Eigen::MatrixXd::Random(10, 1)*0.5;
	Eigen::MatrixXd W2 = Eigen::MatrixXd::Random(10,10)*0.5;
	Eigen::MatrixXd b2 = Eigen::MatrixXd::Random(10, 1)*0.5;
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
	Eigen::MatrixXd Z1 = W1 * X + b1.replicate(1, X.cols());
	Eigen::MatrixXd A1 = ReLU(Z1);
	Eigen::MatrixXd Z2 = W2 * A1 + b2.replicate(1, A1.cols());
	Eigen::MatrixXd A2 = softmax(Z2);
	return {Z1, A1, Z2, A2};
}

Eigen::MatrixXd one_hot(Eigen::MatrixXd Y) {
	int m = Y.size();
	std::cout << "Y.cols " << Y.cols() << std::endl;
	Eigen::MatrixXd one_hot_Y = Eigen::MatrixXd::Zero(m, 10); // because of 10 digits
	Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(10, 10); // because of 10 digits
	for (int i=0; i<m; ++i) {
		if(i==133) // something unexpected is happening to make Y(0,133)=188
			//std::cout << "row " << i << std::endl;
			std::cout << Y(0,i) << "\n\n" << identity.row(Y(0,i)) << std::endl;
		one_hot_Y.row(i) = identity.row(Y(0, i));
	}
	std::cout << "Loop done" << std::endl;
	one_hot_Y = one_hot_Y.transpose();
	return one_hot_Y;
}

Eigen::MatrixXd ReLU_deriv(Eigen::MatrixXd Z) {
	return (Z.array() > 0).cast<double>();
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
backward_prop(Eigen::MatrixXd Z1, Eigen::MatrixXd A1, Eigen::MatrixXd Z2, Eigen::MatrixXd A2, Eigen::MatrixXd W1, Eigen::MatrixXd W2, Eigen::MatrixXd X, Eigen::MatrixXd Y) {
	double m = Y.size();
	Eigen::MatrixXd one_hot_Y = one_hot(Y);
	std::cout << A2.rows() << ' ' << A2.cols() << '\n' << one_hot_Y.rows() << ' ' << one_hot_Y.cols() << std::endl;
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

Eigen::VectorXi get_predictions(Eigen::MatrixXd A2) {
	Eigen::VectorXi argmax0{A2.cols()};
	for (int col = 0; col < A2.cols(); ++col)
		A2.col(col).maxCoeff(&argmax0[col]);
	return argmax0;
}

double get_accuracy(Eigen::VectorXi predictions, Eigen::MatrixXd Y) {
	// std::cout << predictions << Y << std::endl;
	Eigen::VectorXi results = (predictions.array() == Y.array().cast<int>()).cast<int>();
	return results.sum() / Y.size();
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
gradient_descent(Eigen::MatrixXd X, Eigen::MatrixXd Y, double alpha, int iterations) {
	Eigen::MatrixXd W1, b1, W2, b2;
	std::tie(W1, b1, W2, b2) = init_params();
	for (int i=0; i<iterations; ++i) {
		Eigen::MatrixXd Z1, A1, Z2, A2;
		std::tie(Z1, A1, Z2, A2) = forward_prop(W1, b1, W2, b2, X);
		Eigen::MatrixXd dW1, db1, dW2, db2;
		std::tie(dW1, db1, dW2, db2) = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y);
		std::tie(W1, b1, W2, b2) = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha);
		if (i%10==0) {
			std::cerr << "Iteration: " << i << std::endl;
			Eigen::VectorXi predictions = get_predictions(A2);
			std::cerr << get_accuracy(predictions, Y) << std::endl;
		}
	}
	return {W1, b1, W2, b2};
}

int main (int argc, char* argv[]) {
	Eigen::MatrixXd Y_dev, X_dev, Y_train, X_train;
	std::tie(Y_dev, X_dev, Y_train, X_train) = read_data();

	Eigen::MatrixXd W1, b1, W2, b2;
	std::tie(W1, b1, W2, b2) = gradient_descent(X_train, Y_train, 0.10, 500);

	return 0;
}
