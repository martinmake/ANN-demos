#include <iostream>

#include "fcnn.h"
#include "datatypes.h"
#include "config.h"

int main(void)
{
	std::cout << std::endl << "BEGIN FEED-FORWARD DEMO" << std::endl;

	std::cout << std::endl << "CREATING A 3-4-2 RELU-SOFTMAX FULL CONNECTED NEURAL NETWORK" << std::endl;
	Fcnn fcnn(Config::Nn::architecture, Config::Nn::activation_funcs);

	std::cout << std::endl << "SETTING DUMMY WEIGHTS AND BIASES:" << std::endl;
	fcnn.set_weights(Config::Nn::weights);
	show_data(Config::Nn::weights, 2);

	std::cout << std::endl << "INPUTS ARE:" << std::endl;
	show_data(Config::Nn::inputs, 2);

	std::cout << std::endl << "COMPUTING" << std::endl;
	vector outputs = fcnn.compute(Config::Nn::inputs);
	std::cout << "OUTPUTS COMPUTED" << std::endl;

	std::cout << std::endl << "OUTPUTS ARE:" << std::endl;
	show_data(outputs, 2);

	std::cout << std::endl << "END FEED-FORWARD DEMO" << std::endl << std::endl;
	return 0;
}
