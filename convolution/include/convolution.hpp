#ifndef CUDA_RIBARDIERE_WALCAK_CONVOLUTION_HPP
#define CUDA_RIBARDIERE_WALCAK_CONVOLUTION_HPP

#include "common.hpp"
#include <vector>
#include <map>

typedef const std::vector<int>& const_vector_ref;
typedef std::function<const_mat_ptr(const_mat_ref)> convolution_function;

enum Effect {
    FLOU_GAUSSIEN,
    DETECTION_BORD,
};

const_mat_ptr process(const_mat_ref , const Effect&);
const_mat_ptr apply(const_mat_ref, const_vector_ref, float, float);

const_mat_ptr flou_gaussien(const_mat_ref);
const_mat_ptr detection_bord(const_mat_ref);

const std::map<Effect, convolution_function> effect_functions {
        {Effect::FLOU_GAUSSIEN, flou_gaussien},
        {Effect::DETECTION_BORD, detection_bord}
};

#endif
