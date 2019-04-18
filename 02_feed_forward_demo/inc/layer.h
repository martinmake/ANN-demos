#ifndef _FEED_FORWARD_DEMO_LAYER_H_
#define _FEED_FORWARD_DEMO_LAYER_H_

#include "datatypes.h"

class Layer
{
	private:
		uint16_t m_input_count;
		uint16_t m_output_count;
		matrix   m_weights;
		vector   m_inputs;
		vector   m_outputs;
		activation_func_t m_activation_func;

	public:
		Layer(uint16_t input_count, uint16_t output_count, activation_func_t activation_func);
		~Layer();

		vector& forward(const vector& inputs);
		void set_weights(const matrix& weights);
};

#endif
