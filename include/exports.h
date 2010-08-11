/*
	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Library General Public
	License version 2 as published by the Free Software Foundation.

	This library is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
	Library General Public License for more details.

	You should have received a copy of the GNU Library General Public License
	along with this library; see the file COPYING.LIB.  If not, write to
	the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
	Boston, MA 02110-1301, USA.
*/
#ifndef EXPORTS_H
#define EXPORTS_H

#if __CUDA_ARCH__

#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_EXPORT __host__ __device__
#define CUDA_HOST __host__
#define CUDA_DEVICE __device__
#define CUDA_GLOBAL __global__

#elif !defined(__CUDA_ARCH__)

#define CUDA_EXPORT
#define CUDA_HOST
#define CUDA_DEVICE
#define CUDA_GLOBAL

#endif

#endif //EXPORTS_H