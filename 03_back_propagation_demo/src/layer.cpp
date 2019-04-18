#include <iostream>
#include <stdlib.h>

#include "layer.h"
#include "config.h"

Layer::Layer(uint16_t input_count, uint16_t output_count, activation_func_t activation_func)
	: m_input_count(input_count), m_output_count(output_count), m_activation_func(activation_func)
{
	m_inputs  = vector(m_input_count);
	m_outputs = vector(m_output_count);

	m_weights.resize(m_output_count);
	for (uint16_t i = 0; i < m_output_count; i++)
		m_weights[i].resize(m_input_count + 1);
}

Layer::~Layer()
{
}

vector& Layer::forward(const vector& inputs)
{
	for (uint16_t i = 0; i < m_outputs.size(); i++) {
		double sum = m_weights[i][m_input_count];

		for (uint16_t j = 0; j < m_input_count; j++)
			sum += inputs[j] * m_weights[i][j];

		m_outputs[i] = sum;
	}

	m_activation_func(m_outputs);

	return m_outputs;
}

void Layer::set_weights(const matrix& weights)
{
	m_weights = weights;
}

void Layer::set_random_weights()
{
	for (uint16_t i = 0; i < m_output_count; i++) {
		for (uint16_t j = 0; j < m_input_count; j++)
			m_weights[i][j] = generate_random_weight();
		m_weights[i][m_input_count] = generate_random_weight();
	}
}

inline float Layer::generate_random_weight()
{
	float l = Config::Nn::Weights::Random_generation::lower_limit,
	      h = Config::Nn::Weights::Random_generation::upper_limit;

	return (h - l) * ((float) std::rand() / RAND_MAX) + l;
}
