#include <iostream>

#include "fcnn.h"
#include "datatypes.h"
#include "config.h"

int main(void)
{
	matrix dataset;

	std::cout << std::endl << "BEGIN NEURAL NETWORK TRAINING DEMO" << std::endl;

	std::cout << std::endl << "DATA IS THE FAMOUS IRIS FLOWER SET"                                              << std::endl;
	std::cout <<              "THE GOAL IS TO PREDICT SPECIES FROM SEPAL LENGTH, WIDTH AND PETAL LENGTH, WIDTH" << std::endl;
	std::cout <<              "IRIS SETOSA: 0 0 1, IRIS VERSICOLOR: 0 1 0, IRIS VIRGINICA: 1 0 0"              << std::endl;

	std::cout << std::endl << "CREATING " << Config::Nn::Training::Dataset::Fraction::training * 100 << "% TRAINING AND " << Config::Nn::Training::Dataset::Fraction::testing * 100 << "% TEST DATA MATRICES" << std::endl;

	std::cout << std::endl << "CREATING A 4-7-3 RELU-SOFTMAX FULL CONNECTED NEURAL NETWORK" << std::endl;
	Fcnn fcnn(Config::Nn::architecture,
		  Config::Nn::activation_funcs,
		  Config::Nn::activation_funcs_derivs);
 	fcnn.set_random_weights();

	std::cout << std::endl << "SETTING LEARNING RATE: " << Config::Nn::Training::learning_rate << std::endl;
	std::cout              << "SETTING MOMENTUM:      " << Config::Nn::Training::momentum      << std::endl;
	std::cout              << "SETTING MAX EPOCHS:    " << Config::Nn::Training::max_epochs    << std::endl;

	std::cout << std::endl << "LOADING DATASET" << std::endl;
	dataset = Config::Nn::Dataset::load_func();
	show_data(dataset[0], 2);
	show_data(dataset[60], 2);
	show_data(dataset[110], 2);
	std::cout << "..." << std::endl;

	if (Config::Nn::Training::batch_size == 1)
		std::cout << std::endl << "BEGIN TRAINING USING INCREMENTAL BACK-PROPAGATION" << std::endl;
	else if (Config::Nn::Training::batch_size == (uint16_t) -1)
		std::cout << std::endl << "BEGIN TRAINING USING BATCH BACK-PROPAGATION"       << std::endl;
	else
		std::cout << std::endl << "BEGIN TRAINING USING MINI-BATCH BACK-PROPAGATION (" << Config::Nn::Training::batch_size << ')' << std::endl;
	fcnn.train(dataset);
	std::cout << "TRAINING COMPLETE" << std::endl;

	std::cout << std::endl << "FINAL NETWORK WEIGHTS AND BIAS VALUES:" << std::endl;
	fcnn.show_weights(4);

	std::cout << std::endl << "ACCURACY ON TRAINING DATA: " << fcnn.accuracy.training * 100 << '%' << std::endl;
	std::cout              << "ACCURACY ON TEST     DATA: " << fcnn.accuracy.testing  * 100 << '%' << std::endl;

	std::cout << std::endl << "END NEURAL NETWORK TRAINING DEMO" << std::endl << std::endl;
	return 0;
}
