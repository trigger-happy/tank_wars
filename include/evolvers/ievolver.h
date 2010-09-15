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

#ifndef IEVOLVER_H
#define IEVOLVER_H
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include "types.h"
#include "game_core/tank_ai.h"
#include "data_store/data_store.h"

#define NUM_INSTANCES 1024

// top 5% will be elite
#define ELITE_COUNT		(NUM_INSTANCES*0.05f)
// 15% mutation rate
#define MUTATION_RATE	15

template<typename Derived>
class iEvolver{
public:
	/*!
	Initialize the evolver
	*/
	void initialize(const std::string& aidb,
					const std::string& simdb){
		static_cast<Derived*>(this)->initialize_impl();
		m_ds = new DataStore(aidb, simdb);
		
		if(!m_ds->is_ok()){
			std::cerr << "FAILED TO OPEN DB" << std::endl;
		}
		
		m_framecount = 0;
		m_gen_count = 0;
	}
	
	/*!
	Perform cleanup operations
	*/
	void cleanup(){
		static_cast<Derived*>(this)->cleanup_impl();
		delete m_ds;
		m_ds = NULL;
	}
	
	/*!
	Iterate through a single timestep in the simulation.
	\param dt - The delta time in seconds between timesteps
	\note Each call will simulate a single frame in ALL game instances
	*/
	void frame_step(float dt){
		static_cast<Derived*>(this)->frame_step_impl(dt);
		++m_framecount;
	}
	
	/*!
	Retrieve the current game state
	*/
	void retrieve_state(){
		static_cast<Derived*>(this)->retrieve_state_impl();
	}
	
	/*!
	Perform the genetic algorithms to prepare the next generation
	*/
	void evolve_ga(){
		static_cast<Derived*>(this)->evolve_ga_impl();
	}
	
	/*!
	Retrieve the score of the best individual
	*/
	u32 retrieve_highest_score(){
		return static_cast<Derived*>(this)->retrieve_score_impl();
	}
	
	/*!
	Save the genes of all individuals in the current generation
	*/
	void save_genes(const std::string& fname){
		//static_cast<Derived*>(this)->save_best_gene_impl(fname);
		
		for(u32 i = 0; i < NUM_INSTANCES; ++i){
			ai_key aik;
			aik.generation = m_gen_count;
			aik.id = i;
			
			m_ds->save_gene_data(aik, m_ai[m_scoredata[i].first]);
		}
		
		m_framecount = 0;
	}
	
	/*!
	Pre-fitness setup
	*/
	void prepare_game_state(){
		static_cast<Derived*>(this)->prepare_game_state_impl();
		++m_gen_count;
	}
	
	/*!
	Check if all the tanks in the current generation are dead
	*/
	bool is_game_over(){
		return static_cast<Derived*>(this)->is_game_over_impl();
	}

protected:
	// CPU stuff
	std::vector<Physics::PhysRunner::RunnerCore> m_runner;
	std::vector<TankBullet::BulletCollection> m_bullets;
	std::vector<BasicTank::TankCollection> m_tanks;
	std::vector<AI::AI_Core> m_ai;
	
	// backup of initial data for clean slate
	std::vector<Physics::PhysRunner::RunnerCore> m_runner_b;
	std::vector<TankBullet::BulletCollection> m_bullets_b;
	std::vector<BasicTank::TankCollection> m_tanks_b;
	std::vector<AI::AI_Core> m_ai_b;
	
	// frame counter
	u32 m_framecount;

	// score data
	std::vector<std::pair<u32, u32> > m_scoredata;

private:
	// for data storage
	DataStore* m_ds;
	
	// generation count
	u32 m_gen_count;
};

template<typename T>
bool score_sort(const std::pair<T, T>& lhs, const std::pair<T, T>& rhs){
	if(lhs.second > rhs.second){
		return true;
	}if(lhs.second == rhs.second){
		return lhs.first < rhs.first;
	}
	return false;
}

// declare them as templates just because i'm fed up with organizing
// the source code

template<typename T>
void copy_genes(T* dest, T* src){
	for(u32 i = 0; i < MAX_GENE_DATA; ++i){
		dest->gene_accel[i][0] = src->gene_accel[i][0];
		dest->gene_heading[i][0] = src->gene_heading[i][0];
	}
}

template<typename T>
void reproduce(T* child, T* dad, T* mom){
	u32 pos = rand() % MAX_GENE_DATA;
	
	// copy dad's [0,pos) genes
	for(u32 i = 0; i < pos; ++i){
		child->gene_accel[i][0] = dad->gene_accel[i][0];
		child->gene_heading[i][0] = dad->gene_heading[i][0];
	}
	
	// copy mom's [pos, MAX_GENE_DATA) genes
	for(u32 i = pos; i < MAX_GENE_DATA; ++i){
		child->gene_accel[i][0] = mom->gene_accel[i][0];
		child->gene_heading[i][0] = mom->gene_heading[i][0];
	}
}

template<typename T>
void mutate(T* mutant){
	for(int i = 0; i < MAX_GENE_DATA*0.1f; ++i){
		// mutate the thrust value
		u32 pos = rand() % MAX_GENE_DATA;
		AI::AI_Core::gene_type mval = rand() % MAX_THRUST_VALUES;
		mutant->gene_accel[pos][0] = mval;
		
		// mutate the heading value
		pos = rand() % MAX_GENE_DATA;
		mval = rand() % MAX_HEADING_VALUES;
		mutant->gene_heading[pos][0] = mval;
	}
}

#endif //IEVOLVER_H
