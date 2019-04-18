#ifndef _PERCEPTRON_PERCEPTRON_H_
#define _PERCEPTRON_PERCEPTRON_H_

#include <vector>
#include <inttypes.h>

class Perceptron
{
	private:
		uint16_t            m_input_count;
		std::vector<double> m_weights;

	public:
		Perceptron(uint16_t input_count);
		~Perceptron();

		int8_t              compute(const std::vector<double>& inputs);
		std::vector<double> train(const std::vector<std::vector<double>>& training_data);

	private:
		void   initialize_weights();
		int8_t activation(double v);
		void   shuffle(std::vector<uint16_t>& sequence);
		void   update(int8_t computed, int8_t desired, const std::vector<double>& inputs);
		double generate_random_weight();
};

#endif
