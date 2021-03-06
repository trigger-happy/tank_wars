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
#include <boost/timer.hpp>
#include "evolvers/evolver_cpu.h"
#include "evolvers/evolver_gpu.h"

#define MAX_FRAMESTEPS	18000 // 5 minutes
#define TIME_STEP		1000.0f/60.0f // 60 fps

namespace po = boost::program_options;
using namespace std;

static u32 g_randomseed = 0;

template<typename T>
void perform_evolution(iEvolver<T>& evl, const string& fname = "report.dat"){
	// consistent random seed
	srand(g_randomseed);

	u32 num_generations = 0;
	f32 highest_score = 0.0f;
	for(int i = 0; i < MAX_GENERATIONS; ++i){
		boost::timer evol_timer;
		cout << "Running generation " << i << endl;
		// setup the initial game state
 		evl.prepare_game_state();
		
		for(int dist = 1; dist < NUM_DISTANCE_STATES; ++dist){
			for(int locs = 0; locs < NUM_LOCATION_STATES; ++locs){
				for(int vecs = 0; vecs < 1; ++vecs){
// 					cout << "Scenario: " << dist << " " << locs << " " << vecs << endl;
					evl.prepare_game_scenario(dist, locs, vecs);
					
					int j = 0;
					
					for(j = 0; j < MAX_FRAMESTEPS; ++j){
						// perform a frame step
						evl.frame_step(TIME_STEP);
						
						// retrieve the state for debugging purposes
						evl.retrieve_state();
						
						if(evl.is_game_over()){
							break;
						}
					}
					evl.end_game_scenario();
				}
			}
		}
		
		// end this generation
		evl.finalize();
		
		// get the score of the best individual
		highest_score = evl.retrieve_highest_score();
		//highest_score /= MAX_FRAMESTEPS;
		cout << "Generation: " << i << " survived scenarios: "
			<< highest_score  << " score: "
			<< highest_score/((NUM_DISTANCE_STATES)*NUM_LOCATION_STATES*1)
			<< " time: " << evol_timer.elapsed() << endl;
		if(highest_score/((NUM_DISTANCE_STATES)*NUM_LOCATION_STATES*1) >= 0.999f){
			// close to 1.0f
			num_generations = i+1;
			
			// perform the genetic algorithm
			evl.evolve_ga();
			
			// save the data
			evl.save_data(fname);
			
			break;
		}
		
		// perform the genetic algorithm
		evl.evolve_ga();
		
		// save the data
		evl.save_data(fname);
	}
	
	// summary of the evolution process
	cout << "Number of generations: " << num_generations << endl;
	cout << "Best score: " << highest_score << endl;
	
	evl.cleanup();
}

int main(int argc, char* argv[]){
	// seed the random values
	po::options_description desc("Available options");
	desc.add_options()
		("help", "display this help message")
		("cpu", "evolve the AI using the cpu")
		("gpu", "evolve the AI using the gpu")
		("seed", po::value<u32>(), "Use this as the initial random seed");
		
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);
	
	if(vm.count("help")){
		std::cout << desc << std::endl;
		return 0;
	}

	if(vm.count("seed")){
		g_randomseed = vm["seed"].as<u32>();
	}
	
	if(vm.count("cpu")){
		boost::timer t;
		Evolver_cpu evc;
		evc.initialize("genes_cpu.kch", "simulation_cpu.kch");
		perform_evolution(evc);
		cout << "Total evolution time: " << t.elapsed() << endl;
	}else if(vm.count("gpu")){
		boost::timer t;
		Evolver_gpu evg;
		evg.initialize("genes_gpu.kch", "simulation_gpu.kch");
		perform_evolution(evg);
		cout << "Total evolution time: " << t.elapsed() << endl;
	}else{
		// perform both
		cout << "Performing CPU evolution" << endl;
		{
			boost::timer t;
			Evolver_cpu evc;
			evc.initialize("genes_cpu.kch", "simulation_cpu.kch");
			perform_evolution(evc);
			cout << "Total evolution time: " << t.elapsed() << endl;
		}
		cout << "Performing GPU evolution" << endl;
		{
			boost::timer t;
			Evolver_gpu evg;
			evg.initialize("genes_gpu.kch", "simulation_gpu.kch");
			perform_evolution(evg);
			cout << "Total evolution time: " << t.elapsed() << endl;
		}
	}
	
	return 0;
}