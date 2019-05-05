#include <iostream>

#include "perceptron.h"
#include "conf.h"

void show_vector(const std::vector<double> vec, uint8_t precision)
{
	printf("%+1.*f", precision, vec[0]);
	for (uint16_t x = 1; x < vec.size(); x++)
		printf("\n%+1.*f", precision, vec[x]);
	std::cout << std::endl;
}

void show_matrix(const std::vector<std::vector<double>> mat, uint8_t precision)
{
	for (uint16_t y = 0; y < mat.size(); y++) {
		for (uint16_t x = 0; x < mat[y].size(); x++)
			printf("%+1.*f\t", precision, mat[y][x]);
		std::cout << std::endl;
	}
}

int main(void)
{
	std::cout << std::endl << "BEGIN PERCEPTRON DEMO" << std::endl;
	std::cout << std::endl << "PREDICT LIBERAL (-1) OR CONSERVATIVE (+1) FROM AGE, INCOME" << std::endl;

	std::vector<std::vector<double>> training_data = {
		{1.5, 2.0, -1},
		{2.0, 3.5, -1},
		{3.0, 5.0, -1},
		{3.5, 2.5, -1},
		{4.5, 5.0,  1},
		{5.0, 7.0,  1},
		{5.5, 8.0,  1},
		{6.0, 6.0,  1}
	};
	uint16_t input_count = training_data[0].size() - 1;

	std::cout << std::endl << "THE TRAINING DATA IS:" << std::endl;
	show_matrix(training_data, 1);

	std::cout << std::endl << "CREATING PERCEPTRON" << std::endl;
	Perceptron p(input_count);
	std::cout << "SETTING LEARNING RATE TO: " << ALPHA << std::endl;
	std::cout << "SETTING MAX EPOCHS TO:    " << MAX_EPOCHS << std::endl;

	std::cout << std::endl << "BEGIN TRAINING" << std::endl;
	std::vector<double> final_weights = p.train(training_data);
	std::cout << "TRAINING COMPLETE" << std::endl;

	std::cout << std::endl << "BEST WEIGHTS AND BIAS FOUND:" << std::endl;
	show_vector(final_weights, 6);


	std::vector<std::vector<double>> new_data = {
		{3.0, 4.0, -1},
		{0.0, 1.0, -1},
		{2.0, 5.0, -1},
		{5.0, 6.0,  1},
		{9.0, 9.0,  1},
		{4.0, 6.0,  1},
	};

	std::cout << std::endl << "PREDICTION FOR NEW PEOPLE:" << std::endl;
	for (uint16_t i = 0; i < new_data.size(); i++) {
		std::cout << "AGE: " << new_data[i][0] << ", INCOME: " << new_data[i][1];
		int8_t c = p.compute(new_data[i]);
		printf(", PREDICTION: %+1d, CORRECT: %+1d", c, (int) new_data[i][input_count]);
		std::cout << std::endl;
	}

	std::cout << std::endl;

	return 0;
}
