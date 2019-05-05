#include <iostream>
#include <algorithm>
#include <stdlib.h>

#include "fcnn.h"
#include "config.h"

Fcnn::Fcnn(const std::vector<uint16_t>&                architecture,
	   const std::vector<activation_func_t>&       activation_funcs,
	   const std::vector<activation_func_deriv_t>& activation_funcs_derivs)
{
	m_input_count = architecture[0];
	for (uint8_t l = 1; l < architecture.size(); l++) {
		m_layers.push_back(Layer(architecture[l-1],
					 architecture[l],
					 activation_funcs[l-1],
					 activation_funcs_derivs[l-1]));
	}

	m_outputs.resize(architecture.back());

	srand(Config::Nn::Weights::Random_generation::seed);
}

Fcnn::~Fcnn()
{
}

vector& Fcnn::forward(const vector& inputs)
{
	m_outputs = inputs;

	for (uint8_t l = 0; l < m_layers.size(); l++)
		m_outputs = m_layers[l].forward(m_outputs);

	return m_outputs;
}

void Fcnn::backward(const vector& inputs, const vector& target_outputs)
{
	forward(inputs);

 	vector downstream_gradients;
 	m_layers.back().backward_output(target_outputs, downstream_gradients);
 	for (uint8_t l = m_layers.size() - 2; l > 0; l--)
 		m_layers[l].backward_hidden(downstream_gradients);
 	m_layers[0].backward_last(downstream_gradients);
}

void Fcnn::train(matrix& dataset)
{
	std::random_shuffle(dataset.begin(), dataset.end());

	matrix dataset_training(&dataset[0], &dataset[dataset.size() * Config::Nn::Training::Dataset::Fraction::training]),
	       dataset_testing(&dataset[dataset.size() * Config::Nn::Training::Dataset::Fraction::training], &dataset[dataset.size()]);

 	for (uint16_t epoch = 0; epoch <= Config::Nn::Training::max_epochs; epoch++) {
		for (uint16_t d = 0; d < dataset_training.size(); d++) {
			vector inputs        (&dataset_training[d][0], &dataset_training[d][m_input_count]),
			       target_outputs(&dataset_training[d][m_input_count], &dataset_training[d][dataset_training[d].size()]);

			backward(inputs, target_outputs);

		// 	std::cout << "TARGER: ";
 		// 	show_data(target_outputs, 4);
		// 	std::cout << "OUTPUT: ";
 		// 	show_data(m_outputs, 4);

			if (is_correct(target_outputs))
				accuracy.training++;
		}

 	// 	if (epoch % 100 == 0) {
 	// 		printf("EPOCH: %7u\t\tOUTPUTS:\t", epoch);
 	// 		show_data(m_outputs, 4);
 	// 	}
 	}
	accuracy.training /= Config::Nn::Training::max_epochs * dataset_training.size();

 	test(dataset_testing);
}

void Fcnn::test(const matrix& dataset)
{
	for (uint16_t d = 0; d < dataset.size(); d++) {
		vector inputs        (&dataset[d][0], &dataset[d][m_input_count]),
		       target_outputs(&dataset[d][m_input_count], &dataset[d][dataset[d].size()]);

 		forward(inputs);
		if (is_correct(target_outputs))
			accuracy.testing++;
	}

	accuracy.testing /= dataset.size();
}

void Fcnn::set_weights(const tensor& weights)
{
	for (uint8_t l = 0; l < m_layers.size(); l++)
		m_layers[l].set_weights(weights[l]);
}

void Fcnn::set_random_weights()
{
	for (uint8_t l = 0; l < m_layers.size(); l++)
		m_layers[l].set_random_weights();
}

void Fcnn::show_weights(uint8_t precision)
{
	m_layers[0].show_weights(precision);
	for (uint8_t l = 1; l < m_layers.size(); l++) {
		std::cout << std::endl;
		m_layers[l].show_weights(precision);
	}
}


bool Fcnn::is_correct(const vector& desired)
{
	return std::distance(desired.begin(), std::max_element(desired.begin(), desired.end())) == std::distance(m_outputs.begin(), std::max_element(m_outputs.begin(), m_outputs.end()));
}
