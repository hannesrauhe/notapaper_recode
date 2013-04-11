/*
 * CUDAHelper.hpp
 *
 *  Created on: 24.10.2011
 *      Author: fbeier
 */

#ifndef CUDAHELPER_HPP_
#define CUDAHELPER_HPP_

#define  CUDA_SAFE_CALL( err ) CUDAHelper::safeCall( err )

#include "cuda_runtime.h"
#include <iostream>

class CUDAHelper {

public:

	static void safeCall( cudaError_t err ) {
		if ( cudaSuccess != err ) {
			std::cerr << "CUDA Runtime API error(" << err << "): \""
					<< cudaGetErrorString( err ) << "\"" << std::endl;
			exit(1);
		}
	}
};

#endif /* CUDAHELPER_HPP_ */
