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
		tensor weights = {
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
		vector inputs = { 3.3, 1.2, 4.0 };
	}
}
