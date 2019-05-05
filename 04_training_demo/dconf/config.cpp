#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
#include <math.h>

#include "config.h"

inline void relu(vector& sums)
{
	for (uint16_t i = sums.size() - 1; i; i--)
		if (sums[i] < 0)
			sums[i] = 0;
}
inline float relu_deriv(float output)
{
	if (output > 0)
		return 1;
	else
		return 0;
}

inline void softmax(vector& sums)
{
	float max = sums[0];
	for (uint16_t i = 0; i < sums.size(); i++) {
		if (sums[i] > max)
			max = sums[i];
	}

	float scale = 0.0;
	for (uint16_t i = 0; i < sums.size(); i++)
		scale += std::exp(sums[i] - max);

	for (uint16_t i = 0; i < sums.size(); i++)
		sums[i] = std::exp(sums[i] - max) / scale;
}
inline float softmax_deriv(float output)
{
	return output * (1 - output);
}

inline void passtrough(vector& sums)
{
}
inline float passtrough_deriv(float output)
{
	return 1;
}

inline matrix& load_iris(void)
{
	matrix* dataset = new matrix();
	std::ifstream dataset_csv(Config::Nn::Dataset::path);

	for (std::string line; getline(dataset_csv, line); )
	{
		std::stringstream line_stream(line);
		vector data;

		for (uint8_t i = 0; i < 4; i++) {
			std::string cell;
			getline(line_stream, cell, ',');
			data.push_back(stof(cell));
		}
		std::string cell;
		getline(line_stream, cell);
		if (cell == "setosa") {
			data.push_back(0.0);
			data.push_back(0.0);
			data.push_back(1.0);
		} else if (cell == "versicolor") {
			data.push_back(0.0);
			data.push_back(1.0);
			data.push_back(0.0);
		} else if (cell == "virginica") {
			data.push_back(1.0);
			data.push_back(0.0);
			data.push_back(0.0);
		} else {
			std::cout << "ERROR: \"" << cell << "\"IS INVALID SPECIES!)" << std::endl;
			exit(1);
		}

		dataset->push_back(data);
	}

	return *dataset;
}

namespace Config
{
	namespace Nn
	{
		std::vector<uint16_t>                architecture            = { 4,          7,          3             };
		std::vector<activation_func_t>       activation_funcs        = { /* input */ relu,       softmax       };
		std::vector<activation_func_deriv_t> activation_funcs_derivs = { /* input */ relu_deriv, softmax_deriv };

		namespace Training
		{
			float    learning_rate =    0.05;
			float    momentum      =    0.01;
			uint16_t max_epochs    = 1000;
			uint16_t batch_size    =    1;

			namespace Dataset
			{
				namespace Fraction
				{
					float training = 0.8;
					float testing  = 1 - training;
				}
			}
		}

		namespace Dataset
		{
			std::string path = "nn/datasets/iris.csv";
			load_func_t load_func = load_iris;
		}

		namespace Weights
		{
			namespace Random_generation
			{
				uint16_t seed = time(nullptr);

				float lower_limit = -0.01;
				float upper_limit = +0.01;
			}
		}
	}
}
