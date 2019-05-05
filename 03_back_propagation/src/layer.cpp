#include <stdlib.h>
#include <iostream>

#include "layer.h"
#include "config.h"

Layer::Layer(uint16_t                input_count,
	     uint16_t                output_count,
	     activation_func_t       activation_func,
	     activation_func_deriv_t activation_func_deriv)
	: m_input_count          (input_count),
	  m_output_count         (output_count),
	  m_activation_func      (activation_func),
	  m_activation_func_deriv(activation_func_deriv)
{
	m_inputs  = vector(m_input_count);
	m_outputs = vector(m_output_count);

	m_last_deltas.resize(m_output_count);
	for (uint16_t d = 0; d < m_output_count; d++)
		m_last_deltas[d] = vector(m_input_count + 1, 0.0);
}

Layer::~Layer()
{
}

vector& Layer::forward(const vector& inputs)
{
	m_inputs = inputs;

	for (uint16_t p = 0; p < m_outputs.size(); p++) {
		double sum = m_weights[p][m_input_count];

		for (uint16_t i = 0; i < m_input_count; i++)
			sum += m_inputs[i] * m_weights[p][i];

		m_outputs[p] = sum;
	}

	m_activation_func(m_outputs);

	return m_outputs;
}

void Layer::backward_output(const vector& target_outputs, vector& downstream_gradients)
{
	downstream_gradients = vector(m_input_count, 0.0);

	for (uint16_t p = 0; p < m_output_count; p++) {
		float gradient = m_activation_func_deriv(m_outputs[p]) * (target_outputs[p] - m_outputs[p]);

		for (uint16_t i = 0; i < m_input_count; i++) {
			float delta = Config::Nn::learning_rate * gradient * m_inputs[i];
			m_weights[p][i] += delta + Config::Nn::momentum * m_last_deltas[p][i];
			downstream_gradients[i] += m_weights[p][i] * gradient;
			m_last_deltas[p][i] = delta;
		}
		float delta = Config::Nn::learning_rate * gradient;
		m_weights[p][m_input_count] += delta + Config::Nn::momentum * m_last_deltas[p][m_input_count];
		m_last_deltas[p][m_input_count] = delta;
	}
}

void Layer::backward_hidden(vector& downstream_gradients)
{
	for (uint16_t p = 0; p < m_output_count; p++) {
		float gradient = m_activation_func_deriv(m_outputs[p]) * downstream_gradients[p];

		for (uint16_t i = 0; i < m_input_count; i++) {
			float delta = Config::Nn::learning_rate * gradient * m_inputs[i];

			m_weights[p][i] += delta + Config::Nn::momentum * m_last_deltas[p][i];
			downstream_gradients[i] += m_weights[p][i] * gradient;
			m_last_deltas[p][i] = delta;
		}
		float delta = Config::Nn::learning_rate * gradient;
		m_weights[p][m_input_count] += delta + Config::Nn::momentum * m_last_deltas[p][m_input_count];
		m_last_deltas[p][m_input_count] = delta;
	}
}

void Layer::backward_last(const vector& downstream_gradients)
{
	for (uint16_t p = 0; p < m_output_count; p++) {
		float gradient = m_activation_func_deriv(m_outputs[p]) * downstream_gradients[p];

		for (uint16_t i = 0; i < m_input_count; i++) {
			float delta = Config::Nn::learning_rate * gradient * m_inputs[i];

			m_weights[p][i] += delta + Config::Nn::momentum * m_last_deltas[p][i];
			m_last_deltas[p][i] = delta;
		}
		float delta = Config::Nn::learning_rate * gradient;
		m_weights[p][m_input_count] += delta + Config::Nn::momentum * m_last_deltas[p][m_input_count];
		m_last_deltas[p][m_input_count] = delta;
	}
}

void Layer::set_weights(const matrix& weights)
{
	m_weights = weights;
}

void Layer::set_random_weights()
{
	m_weights.resize(m_output_count);
	for (uint16_t p = 0; p < m_output_count; p++) {
		m_weights[p].resize(m_input_count + 1);
		for (uint16_t i = 0; i < m_input_count; i++)
			m_weights[p][i] = generate_random_weight();
		m_weights[p][m_input_count] = generate_random_weight();
	}
}

inline float Layer::generate_random_weight()
{
	float l = Config::Nn::Weights::Random_generation::lower_limit,
	      h = Config::Nn::Weights::Random_generation::upper_limit;

	return (h - l) * ((float) std::rand() / RAND_MAX) + l;
}
