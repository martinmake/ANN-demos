#include <iostream>

#include "fcnn.h"
#include "datatypes.h"
#include "config.h"

int main(void)
{
	std::cout << std::endl << "BEGIN BACK-PROPAGATION DEMO" << std::endl;

	std::cout << std::endl << "CREATING A 3-4-2 RELU-SOFTMAX FULL CONNECTED NEURAL NETWORK" << std::endl;
	Fcnn fcnn(Config::Nn::architecture, Config::Nn::activation_funcs);

	std::cout << std::endl << "SETTING DUMMY WEIGHTS AND BIASES:" << std::endl;
	fcnn.set_weights(Config::Nn::Weights::initial_weights);
	fcnn.show_weights(2);

	std::cout << std::endl << "SETTING FIXED INPUTS:"         << std::endl;
	show_data(Config::Nn::inputs, 2);
	std::cout              << "SETTING FIXED TARGET OUTPUTS:" << std::endl;
	show_data(Config::Nn::target_outputs, 2);

	std::cout << std::endl << "SETTING LEARNING RATE: " << Config::Nn::learning_rate << std::endl;
	std::cout              << "SETTING MOMENTUM:      " << Config::Nn::momentum      << std::endl;
	std::cout              << "SETTING MAX EPOCHS:    " << Config::Nn::max_epochs    << std::endl;

	std::cout << std::endl << "BEST WEIGHTS AND BIAS FOUND:" << std::endl;
	fcnn.show_weights(2);

	std::cout << std::endl << "END FEED-FORWARD DEMO" << std::endl << std::endl;
	return 0;
}
