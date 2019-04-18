#include <stdlib.h>
#include <time.h>

#include "perceptron.h"
#include "conf.h"

Perceptron::Perceptron(uint16_t input_count)
	: m_input_count(input_count)
{
//	std::srand(time(nullptr));
	initialize_weights();
}

Perceptron::~Perceptron()
{
}

void Perceptron::initialize_weights()
{
	m_weights.resize(m_input_count + 1);
	for (uint16_t i = 0; i < m_weights.size(); i++)
		m_weights[i] = generate_random_weight();
}

int8_t Perceptron::compute(const std::vector<double>& inputs)
{
	double sum = m_weights[m_input_count];

	for (uint16_t i = 0; i < m_input_count; i++)
		sum += inputs[i] * m_weights[i];

	return activation(sum);
}

std::vector<double> Perceptron::train(const std::vector<std::vector<double>>& training_data)
{
	uint16_t epoch = 0;
	std::vector<uint16_t> sequence(training_data.size());
	std::vector<double> final_weights(m_weights.size());

	for (uint16_t i = 0; i < sequence.size(); i++)
		sequence[i] = i;

	for (; epoch < MAX_EPOCHS; epoch++) {
		shuffle(sequence);
		for (uint16_t i = 0; i < training_data.size(); i++) {
			uint16_t idx = sequence[i];
			int8_t desired  = (int) training_data[idx][m_input_count];
			int8_t computed = compute(training_data[idx]);
			update(computed, desired, training_data[idx]);
		}
	}

	final_weights = m_weights;
	return final_weights;
}

int8_t Perceptron::activation(double v)
{
	if (v >= 0)
		return +1;
	else
		return -1;
}

void Perceptron::shuffle(std::vector<uint16_t>& sequence)
{
	for (uint16_t i = 0; i < sequence.size(); i++) {
		uint16_t r = std::rand() % sequence.size();
		uint16_t tmp = sequence[r];
		sequence[r] = sequence[i];
		sequence[i] = tmp;
	}
}

void Perceptron::update(int8_t computed, int8_t desired, const std::vector<double>& inputs)
{
	if (computed == desired) return;
	int8_t delta = desired - computed;

	for (uint16_t i = 0; i < m_input_count; i++)
		m_weights[i] += ALPHA * delta * abs(inputs[i]);
	m_weights[m_input_count] += ALPHA * delta;
}

double Perceptron::generate_random_weight()
{
	double l = -0.01,
	       h = +0.01;

	return (h - l) * ((double) std::rand() / RAND_MAX) + l;
}
