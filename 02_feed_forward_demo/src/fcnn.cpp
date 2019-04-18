#include "fcnn.h"
#include "config.h"

Fcnn::Fcnn(const std::vector<uint16_t>& architecture, const std::vector<activation_func_t>& activation_funcs)
{
	m_input_count = architecture[0];
	for (uint8_t i = 1; i < architecture.size(); i++)
		m_layers.push_back(Layer(architecture[i-1], architecture[i], activation_funcs[i-1]));
}

Fcnn::~Fcnn()
{
}

vector Fcnn::compute(const vector& inputs)
{
	vector outputs = inputs;

	for (uint16_t i = 0; i < m_layers.size(); i++)
		outputs = m_layers[i].forward(outputs);

	return outputs;
}

void Fcnn::set_weights(const tensor& weights)
{
	for (uint8_t i = 0; i < weights.size(); i++)
		m_layers[i].set_weights(weights[i]);
}
