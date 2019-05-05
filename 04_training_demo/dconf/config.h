#ifndef _FEED_FORWARD_DEMO_DCONF_CONFIG_H_
#define _FEED_FORWARD_DEMO_DCONF_CONFIG_H_

#include "datatypes.h"

namespace Config
{
	namespace Nn
	{
		extern std::vector<uint16_t>                architecture;
		extern std::vector<activation_func_t>       activation_funcs;
		extern std::vector<activation_func_deriv_t> activation_funcs_derivs;

		namespace Training
		{
			extern float    learning_rate;
			extern float    momentum;
			extern uint16_t max_epochs;
			extern uint16_t batch_size;

			namespace Dataset
			{
				namespace Fraction
				{
					extern float training;
					extern float testing;
				}
			}
		}

		namespace Dataset
		{
			extern std::string path;
			extern load_func_t load_func;
		}

		namespace Weights
		{
			namespace Random_generation
			{
				extern uint16_t seed;

				extern float lower_limit;
				extern float upper_limit;
			}
		}
	}
}

#endif
