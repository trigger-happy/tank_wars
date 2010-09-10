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
#include <cmath>
#include <boost/program_options.hpp>
#include "evolvers/evolver_cpu.h"
#include "evolvers/evolver_gpu.h"

#define MAX_GENERATIONS	100000
#define MAX_FRAMESTEPS	18000 // 5 minutes
#define TIME_STEP		1000.0f/60.0f // 60 fps

namespace po = boost::program_options;
using namespace std;

template<typename T>
void perform_evolution(iEvolver<T>& evl, const string& fname = "report.dat"){
	evl.initialize();
	
	//TODO: change this to the appropriate terminating condition
	// terminating condition will be the score of the best individual in the
	// latest generation
	u32 num_generations = 0;
	f32 highest_score = 0.0f;
	for(int i = 0; i < MAX_GENERATIONS; ++i){
		cout << "Running generation " << i << endl;
		// setup the initial game state
		evl.prepare_game_state();
		
		for(int j = 0; j < MAX_FRAMESTEPS; ++j){
			// perform a frame step
			evl.frame_step(TIME_STEP);
			
			// retrieve the state for debugging purposes
			evl.retrieve_state();
			
			if(evl.is_game_over()){
				break;
			}
		}
		
		// get the score of the best individual
		highest_score = evl.retrieve_highest_score();
		highest_score /= MAX_FRAMESTEPS;
		cout << "Generation " << i << " score is " << highest_score << endl;
		if(highest_score >= 0.999f){
			// close to 1.0f
			num_generations = i+1;
			break;
		}
		
		// save the data
		evl.save_best_gene(fname);
		
		// perform the genetic algorithm
		evl.evolve_ga();
	}
	
	// summary of the evolution process
	cout << "Number of generations: " << num_generations << endl;
	cout << "Best score: " << highest_score << endl;
	
	evl.cleanup();
}

int main(int argc, char* argv[]){
	// seed the random values
	srand(std::time(NULL));
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