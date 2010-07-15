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
#include <iostream>
#include <vector>
#include <string>
#include "cuda.h"

using namespace std;

void print_details(const cudaDeviceProp& dev_prop){
	cout << "Device Name: " << dev_prop.name << endl;
	cout << "Total Global Mem: " << dev_prop.totalGlobalMem << " bytes" << endl;
	cout << "Max shared mem per block: " << dev_prop.sharedMemPerBlock << " bytes" << endl;
	cout << "Registers per block: " << dev_prop.regsPerBlock << endl;
	cout << "Warp size: " << dev_prop.warpSize << " threads" << endl;
	cout << "Maximum memory pitch: " << dev_prop.memPitch << " bytes" << endl;
	cout << "Maximum threads per block: " << dev_prop.maxThreadsPerBlock
		<< " threads\n" << endl;
	
	cout << "Max block dimensions:\n"
		<< "X: " << dev_prop.maxThreadsDim[0] << " threads\n"
		<< "Y: " << dev_prop.maxThreadsDim[1] << " threads\n"
		<< "Z: " << dev_prop.maxThreadsDim[2] << " threads\n" << endl;
		
	cout << "Max grid dimensions:\n"
		<< "X: " << dev_prop.maxGridSize[0] << " blocks\n"
		<< "Y: " << dev_prop.maxGridSize[1] << " blocks\n"
		<< "Z: " << dev_prop.maxGridSize[2] << " blocks\n" << endl;
		
	cout << "Clock frequency: " << dev_prop.clockRate << " kilohertz" << endl;
	cout << "Total const mem: " << dev_prop.totalConstMem << " bytes" << endl;
	cout << "Compute Capability: " << dev_prop.major << '.' << dev_prop.minor
		<< endl;
	cout << "Device overlap: " << dev_prop.deviceOverlap << endl;
	cout << "Device Multiprocessors: " << dev_prop.multiProcessorCount
		<< " multiprocessors" << endl;
		
	cout << "Kernel Exec time limit: " << dev_prop.kernelExecTimeoutEnabled
		<< endl;
	
	cout << "Integrated: " << dev_prop.integrated << endl;
	cout << "Can map host memory: " << dev_prop.canMapHostMemory << endl;
}

int main(int argc, char* argv[]){
	int dev_count = 0;
	cudaGetDeviceCount(&dev_count);
	
	cout << "CUDA devices found: " << dev_count << endl;
	
	cudaDeviceProp temp;
	cudaError_t err;
	for(int i = 0; i < dev_count; ++i){
		err = cudaGetDeviceProperties(&temp, i);
		if(err == cudaSuccess){
			print_details(temp);
		}
	}
	return 0;
}