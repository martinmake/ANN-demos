#include <iostream>
#include <stdlib.h>

#include "fcnn.h"
#include "config.h"

Fcnn::Fcnn(const std::vector<uint16_t>&                architecture,
	   const std::vector<activation_func_t>&       activation_funcs,
	   const std::vector<activation_func_deriv_t>& activation_funcs_derivs)
{
	m_input_count = architecture[0];
	for (uint8_t i = 1; i < architecture.size(); i++)
		m_layers.push_back(Layer(architecture[i-1],
					 architecture[i],
					 activation_funcs[i-1],
					 activation_funcs_derivs[i-1]));

	srand(Config::Nn::Weights::Random_generation::seed);
}

Fcnn::~Fcnn()
{
}

vector Fcnn::forward(const vector& inputs)
{
	vector outputs = inputs;

	for (uint16_t i = 0; i < m_layers.size(); i++)
		outputs = m_layers[i].forward(outputs);

	return outputs;
}

vector Fcnn::backward(const vector& inputs)
{
}

void Fcnn::learn(const vector& target_outputs, const vector& inputs)
{
}

void Fcnn::set_weights(const tensor& weights)
{
	for (uint8_t i = 0; i < weights.size(); i++)
		m_layers[i].set_weights(weights[i]);
}

void Fcnn::set_random_weights()
{
	for (uint8_t i = 0; i < m_layers.size(); i++)
		m_layers[i].set_random_weights();
}

void Fcnn::show_weights(uint8_t precision)
{
	m_layers[0].show_weights(precision);
	for (uint8_t i = 1; i < m_layers.size(); i++) {
		std::cout << std::endl;
		m_layers[i].show_weights(precision);
	}
}
