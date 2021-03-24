//
// Created by Elias Fernandez on 2019-05-21.
//

#ifndef EGTTOOLS_OPENMPUTILS_HPP
#define EGTTOOLS_OPENMPUTILS_HPP

#include <omp.h>
#include <egttools/Types.h>

#pragma omp declare reduction (+: egttools::Vector: omp_out=omp_out+omp_in)\
     initializer(omp_priv=egttools::Vector::Zero(omp_orig.size()))

#pragma omp declare reduction (+: egttools::VectorXui: omp_out=omp_out+omp_in)\
     initializer(omp_priv=egttools::VectorXui::Zero(omp_orig.size()))

#pragma omp declare reduction (+: egttools::VectorXi: omp_out=omp_out+omp_in)\
     initializer(omp_priv=egttools::VectorXi::Zero(omp_orig.size()))

#pragma omp declare reduction (+: egttools::Vector3d: omp_out=omp_out+omp_in)\
     initializer(omp_priv=egttools::Vector3d::Zero())

#endif //DYRWIN_OPENMPUTILS_HPP
