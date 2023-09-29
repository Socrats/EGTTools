/** Copyright (c) 2019-2021  Elias Fernandez
  *
  * This file is part of EGTtools.
  *
  * EGTtools is free software: you can redistribute it and/or modify
  * it under the terms of the GNU General Public License as published by
  * the Free Software Foundation, either version 3 of the License, or
  * (at your option) any later version.
  *
  * EGTtools is distributed in the hope that it will be useful,
  * but WITHOUT ANY WARRANTY; without even the implied warranty of
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  * GNU General Public License for more details.
  *
  * You should have received a copy of the GNU General Public License
  * along with EGTtools.  If not, see <http://www.gnu.org/licenses/>
*/

#pragma once
#ifndef EGTTOOLS_OPENMPUTILS_HPP
#define EGTTOOLS_OPENMPUTILS_HPP

#include <egttools/Types.h>

#pragma omp declare reduction (+: egttools::Vector: omp_out=omp_out+omp_in)\
     initializer(omp_priv=egttools::Vector::Zero(omp_orig.size()))

#pragma omp declare reduction (+: egttools::VectorXui: omp_out=omp_out+omp_in)\
     initializer(omp_priv=egttools::VectorXui::Zero(omp_orig.size()))

#pragma omp declare reduction (+: egttools::VectorXi: omp_out=omp_out+omp_in)\
     initializer(omp_priv=egttools::VectorXi::Zero(omp_orig.size()))

#pragma omp declare reduction (+: egttools::Vector3d: omp_out=omp_out+omp_in)\
     initializer(omp_priv=egttools::Vector3d::Zero(omp_orig.size()))

#pragma omp declare reduction (+: egttools::Matrix2D: omp_out=omp_out+omp_in)\
     initializer(omp_priv=egttools::Matrix2D::Zero(omp_orig.rows(), omp_orig.cols()))

#pragma omp declare reduction (+: egttools::SparseMatrix2DXui: omp_out=omp_out+omp_in)\
     initializer(omp_priv=egttools::SparseMatrix2DXui(omp_orig.rows(), omp_orig.cols()))

#endif //EGTTOOLS_OPENMPUTILS_HPP
