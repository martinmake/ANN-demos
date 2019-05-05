#include <time.h>
#include <math.h>

#include "config.h"

inline void relu(vector& sums)
{
	for (uint16_t i = sums.size() - 1; i; i--)
		if (sums[i] < 0)
			sums[i] = 0;
}
inline float relu_deriv(float output)
{
	if (output > 0)
		return 1;
	else
		return 0;
}

inline void softmax(vector& sums)
{
	float max = sums[0];
	for (uint16_t i = 0; i < sums.size(); i++) {
		if (sums[i] > max)
			max = sums[i];
	}

	float scale = 0.0;
	for (uint16_t i = 0; i < sums.size(); i++)
		scale += std::exp(sums[i] - max);

	for (uint16_t i = 0; i < sums.size(); i++)
		sums[i] = std::exp(sums[i] - max) / scale;
}
inline float softmax_deriv(float output)
{
	return output * (1 - output);
}

inline void passtrough(vector& sums)
{
}
inline float passtrough_deriv(float output)
{
	return 1;
}

namespace Config
{
	namespace Nn
	{
		std::vector<uint16_t>                architecture            = { 3,          4,               2        };
		std::vector<activation_func_t>       activation_funcs        = { /* input */ relu,            softmax  };
		std::vector<activation_func_deriv_t> activation_funcs_derivs = { /* input */ relu_deriv, softmax_deriv };

		float    learning_rate = 0.05;
		float    momentum      = 0.01;
		uint16_t max_epochs    = 1000;

		vector inputs         = { 1.0, 2.0, 3.0 };
		vector target_outputs = { 0.25, 0.75 };

		namespace Weights
		{
			tensor initial_weights =
			{
				{
					{ 0.01, 0.02, 0.03, 0.04 },
					{ 0.05, 0.06, 0.07, 0.08 },
					{ 0.09, 0.10, 0.11, 0.12 },
					{ 0.13, 0.14, 0.15, 0.16 }
				},
				{
					{ 0.17, 0.18, 0.19, 0.20, 0.21 },
					{ 0.22, 0.23, 0.24, 0.25, 0.26 }
				}
			};

			namespace Random_generation
			{
				uint16_t seed = time(nullptr);

				float lower_limit = -0.01;
				float upper_limit = +0.01;
			}
		}
	}
}
