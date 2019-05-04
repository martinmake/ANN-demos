#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>

#define DATASET_TEXT "dataset.csv"
#define DATASET_BIN  "dataset.bin"

int main(void)
{
	std::ifstream dataset_text(DATASET_TEXT);
	std::ofstream dataset_bin (DATASET_BIN, std::ios::trunc | std::ios::binary);

	for (std::string line; std::getline(dataset_text, line) && !line.empty(); ) {
		std::stringstream line_stream(line);
		std::string cell;

		for (uint8_t i = 0; i < 4; i++) {
			std::getline(line_stream, cell, ',');
			float val = stof(cell);
			dataset_bin.write((char *) &val, sizeof(float));
		}

		std::string species;
		std::getline(line_stream, species);
		float vals[3];
		memset(vals, 0, sizeof(vals));
		if (species == "setosa")
			vals[2] = 1.0;
		else if (species == "versicolor")
			vals[1] = 1.0;
		else
			vals[0] = 1.0;
		dataset_bin.write((char *) vals, sizeof(vals));
	}

	dataset_text.close();
	dataset_bin.close();

	return 0;
}
