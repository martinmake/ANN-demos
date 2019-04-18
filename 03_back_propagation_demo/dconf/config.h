#ifndef _FEED_FORWARD_DEMO_DCONF_CONFIG_H_
#define _FEED_FORWARD_DEMO_DCONF_CONFIG_H_

#include "datatypes.h"

namespace Config
{
	namespace Nn
	{
		extern std::vector<uint16_t> architecture;
		extern std::vector<activation_func_t> activation_funcs;

		extern float    learning_rate;
		extern float    momentum;
		extern uint16_t max_epochs;

		extern vector inputs;
		extern vector target_outputs;

		namespace Weights
		{
			extern tensor initial_weights;
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
