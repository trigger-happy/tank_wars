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
#include <string>
#include <vector>
#include <boost/program_options.hpp>
#include "evolvers/evolver_cpu.h"
#include "evolvers/evolver_gpu.h"

#define MAX_GENERATIONS	10
#define MAX_FRAMESTEPS	18000 // 5 minutes
#define TIME_STEP		0.0166666667 // 60 fps

namespace po = boost::program_options;

template<typename T>
void perform_evolution(iEvolver<T>& evl){
	evl.initialize();
	
	//TODO: change this to the appropriate terminating condition
	// terminating condition will be the score of the best individual in the
	// latest generation
	for(int i = 0; i < MAX_GENERATIONS; ++i){
		for(int j = 0; j < MAX_FRAMESTEPS; ++j){
			evl.frame_step(TIME_STEP);
			evl.retrieve_state();
		}
		//TODO: perform some GA magic here
	}
	
	evl.cleanup();
}

int main(int argc, char* argv[]){
	po::options_description desc("Available options");
	desc.add_options()
		("help", "display this help message")
		("cpu", "evolve the AI using the cpu")
		("gpu", "evolve the AI using the gpu");
		
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);
	
	if(vm.count("help")){
		std::cout << desc << std::endl;
		return 0;
	}
	
	if(vm.count("cpu")){
		Evolver_cpu evc;
		perform_evolution(evc);
	}else if(vm.count("gpu")){
		Evolver_gpu evg;
		perform_evolution(evg);
	}
	
	return 0;
}