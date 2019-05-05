#ifndef _FEED_FORWARD_DEMO_DATATYPES_H_
#define _FEED_FORWARD_DEMO_DATATYPES_H_

#include <vector>
#include <string>
#include <inttypes.h>

using vector = std::vector<float>;
using matrix = std::vector<std::vector<float>>;
using tensor = std::vector<std::vector<std::vector<float>>>;

using activation_func_t       = void  (*)(vector& sums);
using activation_func_deriv_t = float (*)(float outputs);

using load_func_t = matrix& (*)(void);

template <typename T>
extern void show_data(const T& data, uint8_t precision);

#endif
