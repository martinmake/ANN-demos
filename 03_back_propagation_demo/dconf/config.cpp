#include <time.h>

#include "config.h"

void relu(vector& sums)
{
	for (uint16_t i = sums.size() - 1; i; i--)
		if (sums[i] < 0)
			sums[i] = 0;
}

void softmax(vector& sums) // WIP
{
}

void passtrough(vector& sums)
{
}

namespace Config
{
	namespace Nn
	{
		std::vector<uint16_t> architecture              = { 3, 4   , 2       };
		std::vector<activation_func_t> activation_funcs = {    relu, softmax };

		float    learning_rate = 0.05;
		float    momentum      = 0.01;
		uint16_t max_epochs    = 1000;

		vector inputs         = { 1.0, 2.0, 3.0 };
		vector target_outputs = { 0.25, 0.75 };

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
