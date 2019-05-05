#ifndef _ANNDEMOS_TRAINING_FCNN_H_
#define _ANNDEMOS_TRAINING_FCNN_H_

#include "datatypes.h"
#include "layer.h"

struct Accuracy
{
	float training;
	float testing;
};

class Fcnn
{
	private:
		uint16_t m_input_count;
		vector m_outputs;
		std::vector<Layer> m_layers;

	public:
		Accuracy accuracy;

	public:
		Fcnn(const std::vector<uint16_t>&                architecture,
		     const std::vector<activation_func_t>&       activation_funcs,
		     const std::vector<activation_func_deriv_t>& activation_funcs_derivs);
		~Fcnn();

		vector& forward(const vector& inputs);
		void backward(const vector& inputs, const vector& target_outputs);
		void train(matrix& dataset);
		void set_weights(const tensor& weights);
		void set_random_weights();
		void show_weights(uint8_t precision, uint8_t min_digits);

	private:
		void test(const matrix& dataset);
		bool is_correct(const vector& desired);
};

#endif
