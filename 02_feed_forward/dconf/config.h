#ifndef _FEED_FORWARD_DEMO_DCONF_CONFIG_H_
#define _FEED_FORWARD_DEMO_DCONF_CONFIG_H_

#include "datatypes.h"

namespace Config
{
	namespace Nn
	{
		extern std::vector<uint16_t> architecture;
		extern std::vector<activation_func_t> activation_funcs;
		extern tensor weights;
		extern vector inputs;
	}
}

#endif
