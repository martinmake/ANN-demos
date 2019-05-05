#include "iostream"
#include "datatypes.h"

template <>
void show_data(const vector& data, uint8_t precision)
{
	for (uint16_t x = 0; x < data.size(); x++)
		printf("%+1.*f\t", precision, data[x]);
	std::cout << std::endl;
}

template <>
void show_data(const matrix& data, uint8_t precision)
{
	for (uint16_t i = 0; i < data.size(); i++)
		show_data(data[i], precision);
}

template <>
void show_data(const tensor& data, uint8_t precision)
{
	show_data(data[0], precision);
	for (uint16_t i = 1; i < data.size(); i++) {
		std::cout << std::endl;
		show_data(data[i], precision);
	}
}
