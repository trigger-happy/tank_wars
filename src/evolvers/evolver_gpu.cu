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
#include <fstream>
#include <algorithm>
#include <cuda.h>
#include "exports.h"
#include "evolvers/evolver_gpu.h"
#include "../game_core/twgame.cu"

#define CUDA_BLOCKS		NUM_INSTANCES
#define CUDA_THREADS	MAX_ARRAY_SIZE

// top 15% will be elite
#define ELITE_COUNT		(NUM_INSTANCES*0.15f)
// 25% mutation rate
#define MUTATION_RATE	25

using namespace std;

void Evolver_gpu::initialize_impl(){
	m_framecount = 0;
	// resize the vectors
	m_runner.resize(NUM_INSTANCES);
	m_bullets.resize(NUM_INSTANCES);
	m_tanks.resize(NUM_INSTANCES);
	m_ai.resize(NUM_INSTANCES);
	
	// setup everything on the CPU
	for(u32 i = 0; i < NUM_INSTANCES; ++i){
		Physics::PhysRunner::initialize(&m_runner[i]);
		TankBullet::initialize(&m_bullets[i], &m_runner[i]);
		BasicTank::initialize(&m_tanks[i], &m_runner[i], &m_bullets[i]);
		AI::initialize(&m_ai[i], &m_tanks[i], &m_bullets[i]);
	}
	
	// create a backup copy for quick re-initialization
	m_ai_b = m_ai;
	m_runner_b = m_runner;
	m_tanks_b = m_tanks;
	m_bullets_b = m_bullets;
	
	// now we start up the GPU stuff
	cudaMalloc(reinterpret_cast<void**>(&m_cuda_runner),
				NUM_INSTANCES*sizeof(Physics::PhysRunner::RunnerCore));
	cudaMalloc(reinterpret_cast<void**>(&m_cuda_bullets),
				NUM_INSTANCES*sizeof(TankBullet::BulletCollection));
	cudaMalloc(reinterpret_cast<void**>(&m_cuda_tanks),
				NUM_INSTANCES*sizeof(BasicTank::TankCollection));
	cudaMalloc(reinterpret_cast<void**>(&m_cuda_ai),
				NUM_INSTANCES*sizeof(AI::AI_Core));
				
	// copy over the stuff to GPU mem
	copy_to_device();
}

void Evolver_gpu::cleanup_impl(){
	cudaFree(m_cuda_tanks);
	cudaFree(m_cuda_bullets);
	cudaFree(m_cuda_runner);
	cudaFree(m_cuda_ai);
}

__global__ void internal_frame_step(f32 dt,
									Physics::PhysRunner::RunnerCore* runners,
									TankBullet::BulletCollection* bullets,
									BasicTank::TankCollection* tanks,
									AI::AI_Core* aic){
	// perform all the AI operations
	AI::timestep(&aic[blockIdx.x], dt);
	
	// perform the physics operations
	Physics::PhysRunner::timestep(&runners[blockIdx.x], dt);
	
	// update the bullets
	TankBullet::update(&bullets[blockIdx.x], dt);
	
	// update the tanks
	BasicTank::update(&tanks[blockIdx.x], dt);
	
	// collision check
	if(threadIdx.x < MAX_BULLETS){
		Collision::bullet_tank_check(&bullets[blockIdx.x],
									 &tanks[blockIdx.x],
									 threadIdx.x);
	}
	if(threadIdx.x < MAX_TANKS){
		Collision::tank_tank_check(&tanks[blockIdx.x],
								   threadIdx.x);
	}
}

void Evolver_gpu::frame_step_impl(f32 dt){
	internal_frame_step<<<CUDA_BLOCKS, CUDA_THREADS>>>(dt,
													   m_cuda_runner,
													   m_cuda_bullets,
													   m_cuda_tanks,
													   m_cuda_ai);
	++m_framecount;
	// CPU based
	// force it
// 	copy_from_device();
// 	for(int i = 0; i < NUM_INSTANCES; ++i){
// 		AI::timestep(&m_ai[i], dt);
// 		Physics::PhysRunner::timestep(&m_runner[i], dt);
// 		TankBullet::update(&m_bullets[i], dt);
// 		BasicTank::update(&m_tanks[i], dt);
// 	}
// 	copy_to_device();
}

void Evolver_gpu::retrieve_state_impl(){
	copy_from_device();
}

template<typename T>
bool score_sort(const pair<T, T>& lhs, const pair<T, T>& rhs){
	if(lhs.second > rhs.second){
		return true;
	}
	return false;
}

void copy_genes(AI::AI_Core* dest, AI::AI_Core* src){
	for(u32 i = 0; i < MAX_GENE_DATA; ++i){
		dest->gene_accel[i][0] = src->gene_accel[i][0];
		dest->gene_heading[i][0] = src->gene_heading[i][0];
	}
}

void reproduce(AI::AI_Core* child, AI::AI_Core* dad, AI::AI_Core* mom){
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

void mutate(AI::AI_Core* mutant){
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

void Evolver_gpu::evolve_ga_impl(){
	// copy over the data to the vector for easy sorting
	vector<pair<u32, u32> > score_data(m_population_score.size());
	for(u32 i = 0; i < score_data.size(); ++i){
		score_data[i].first = i;
		score_data[i].second = m_population_score[i];
	}
	
	// sort it from highest score to lowest score
	sort(score_data.begin(), score_data.end(), score_sort<u32>);

	// perform the reproduction process
	// score_data[n].first is the index to the individual
	// second is the score
	vector<AI::AI_Core> next_gen = m_ai;
	for(u32 i = 0; i < score_data.size(); ++i){
		if(i < ELITE_COUNT){
			// copy over the elite genes
			copy_genes(&next_gen[i], &m_ai[score_data[i].first]);
		}else{
			// time to reproduce given whatever else there may be
			// we'll force only the 1st half of the set of parents
			u32 p1 = rand() % score_data.size()/2;
			u32 p2 = rand() % score_data.size()/2;
			reproduce(&next_gen[i], &m_ai[score_data[p1].first],
					  &m_ai[score_data[p2].first]);

			// random chance to mutate
			u32 m = rand() % 100;
			if(m < MUTATION_RATE){
				mutate(&next_gen[i]);
			}
		}
	}
}

u32 Evolver_gpu::retrieve_score_impl(){
	score_map::iterator best_pos;
	best_pos = std::max_element(m_population_score.begin(),
						   m_population_score.end());
	return best_pos->second;
}

void Evolver_gpu::save_best_gene_impl(const string& fname){
	// make sure that we have the stuff from the GPU
	copy_from_device();

	// find the individual with the highest score
	score_map::iterator best_pos;
	best_pos = max_element(m_population_score.begin(),
						   m_population_score.end());
	
	ofstream fout(fname.c_str());
// 	fout.seekp(ios::end);

	// assume that we just want AI_CONTROLLER 0
	// write out the accel gene 1st
	if(fout.is_open()){
		u32 index = best_pos->first;
		for(int i = 0; i < MAX_GENE_DATA; ++i){
			fout << m_ai[index].gene_accel[i][0];
		}

		// write out the heading gene next
		for(int i = 0; i < MAX_GENE_DATA; ++i){
			fout << m_ai[index].gene_heading[i][0];
		}
	}
	
	fout.close();
}

void Evolver_gpu::prepare_game_state_impl(){
	m_population_score.clear();
	m_framecount = 0;
	// get the backup buffer and put it into current one
	//TODO: figure out how to save the genetic data
	m_ai = m_ai_b;
	m_runner = m_runner_b;
	m_tanks = m_tanks_b;
	m_bullets = m_bullets_b;
	
	// setup stuff on the current buffer
	Physics::vec2 params;
	for(int i = 0; i < NUM_INSTANCES; ++i){
		params.x = -25;
		tank_id evading_tank = BasicTank::spawn_tank(&m_tanks[i],
													 params,
													 0,
													 0);
		params.x = 15;
		tank_id attacking_tank = BasicTank::spawn_tank(&m_tanks[i],
													   params,
													   180,
													   1);
		AI::add_tank(&m_ai[i], evading_tank, AI_TYPE_EVADER);
		AI::add_tank(&m_ai[i], attacking_tank, AI_TYPE_ATTACKER);
	}
	
	// send it over to the device
	copy_to_device();
}

bool Evolver_gpu::is_game_over_impl(){
	//NOTE: we'll also use this function to save their score if needed
	bool all_dead = true;
	for(int i = 0; i < NUM_INSTANCES; ++i){
		// tank 0 is the one dodging, check its status
		all_dead &= (m_tanks[i].state[0] == TANK_STATE_INACTIVE);
		if(m_tanks[i].state[0] == TANK_STATE_INACTIVE
			&& (m_population_score.find(i) == m_population_score.end())){
			// this tank is dead, save the score
			m_population_score[i] = m_framecount;
		}
	}
	return all_dead;
}

void Evolver_gpu::copy_to_device(){
	for(u32 i = 0; i < NUM_INSTANCES; ++i){
		// reset the pointers
		BasicTank::reset_pointers(&m_tanks[i],
								  m_cuda_runner+i,
								  m_cuda_bullets+i);
		TankBullet::reset_phys_pointer(&m_bullets[i],
									   m_cuda_runner+i);
					
		m_ai[i].bc = m_cuda_bullets + i;
		m_ai[i].tc = m_cuda_tanks + i;
	}
	
	cudaMemcpy(m_cuda_runner, &m_runner[0],
				NUM_INSTANCES*sizeof(Physics::PhysRunner::RunnerCore),
				cudaMemcpyHostToDevice);
				
	cudaMemcpy(m_cuda_bullets, &m_bullets[0],
				NUM_INSTANCES*sizeof(TankBullet::BulletCollection),
				cudaMemcpyHostToDevice);
				
	cudaMemcpy(m_cuda_tanks, &m_tanks[0],
				NUM_INSTANCES*sizeof(BasicTank::TankCollection),
				cudaMemcpyHostToDevice);
	
	cudaMemcpy(m_cuda_ai, &m_ai[0],
				NUM_INSTANCES*sizeof(AI::AI_Core), cudaMemcpyHostToDevice);
}

void Evolver_gpu::copy_from_device(){
	cudaMemcpy(&m_bullets[0], m_cuda_bullets,
			   NUM_INSTANCES*sizeof(TankBullet::BulletCollection),
			   cudaMemcpyDeviceToHost);
	cudaMemcpy(&m_tanks[0], m_cuda_tanks,
			   NUM_INSTANCES*sizeof(BasicTank::TankCollection),
			   cudaMemcpyDeviceToHost);
	cudaMemcpy(&m_runner[0], m_cuda_runner,
			   NUM_INSTANCES*sizeof(Physics::PhysRunner::RunnerCore),
			   cudaMemcpyDeviceToHost);
	cudaMemcpy(&m_ai[0], m_cuda_ai,
			   NUM_INSTANCES*sizeof(AI::AI_Core),
			   cudaMemcpyDeviceToHost);
			   
	for(u32 i = 0; i < NUM_INSTANCES; ++i){
		// reset the pointers
		BasicTank::reset_pointers(&m_tanks[i],
								  &m_runner[i],
								  &m_bullets[i]);
		TankBullet::reset_phys_pointer(&m_bullets[i],
									   &m_runner[i]);
					
		m_ai[i].bc = &m_bullets[i];
		m_ai[i].tc = &m_tanks[i];
	}
}
