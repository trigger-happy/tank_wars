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
#include <fstream>
#include <algorithm>
#include <iostream>
#include <cuda.h>
#include "exports.h"
#include "evolvers/evolver_gpu.h"
#include "../game_core/twgame.cu"

#define CUDA_BLOCKS		NUM_INSTANCES
#define CUDA_THREADS	MAX_ARRAY_SIZE
#define RETRIEVE_INTERVAL	60

using namespace std;

void Evolver_gpu::initialize_impl(){
	// resize the vectors
	m_runner.resize(NUM_INSTANCES);
	m_bullets.resize(NUM_INSTANCES);
	m_tanks.resize(NUM_INSTANCES);
	m_ai.resize(NUM_INSTANCES);
	m_score.resize(NUM_INSTANCES, 0);
	
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
	cudaMalloc(reinterpret_cast<void**>(&m_cuda_score),
			   NUM_INSTANCES*sizeof(u32));
				
	// copy over the stuff to GPU mem
	copy_to_device();
}

void Evolver_gpu::cleanup_impl(){
	cudaFree(m_cuda_tanks);
	cudaFree(m_cuda_bullets);
	cudaFree(m_cuda_runner);
	cudaFree(m_cuda_ai);
	cudaFree(m_cuda_score);
}

__global__ void internal_frame_step(f32 dt,
									Physics::PhysRunner::RunnerCore* runners,
									TankBullet::BulletCollection* bullets,
									BasicTank::TankCollection* tanks,
									AI::AI_Core* aic,
									u32* scores){
	// perform all the AI operations
	__syncthreads();
	AI::timestep(&aic[blockIdx.x], dt);
	
	// perform the physics operations
	Physics::PhysRunner::timestep(&runners[blockIdx.x], dt);
	
	// update the bullets
	__syncthreads();
	TankBullet::update(&bullets[blockIdx.x], dt);
	
	// update the tanks
	__syncthreads();
	BasicTank::update(&tanks[blockIdx.x], dt);
	
	// collision check
	__syncthreads();
	if(threadIdx.x < MAX_BULLETS){
		Collision::bullet_tank_check(&bullets[blockIdx.x],
									 &tanks[blockIdx.x],
									 threadIdx.x);
	}
	__syncthreads();
	if(threadIdx.x < MAX_TANKS){
		Collision::tank_tank_check(&tanks[blockIdx.x],
								   threadIdx.x);
	}
	__syncthreads();
	if(tanks[blockIdx.x].state[0] != TANK_STATE_INACTIVE
		&& threadIdx.x == 0){
		//scores[blockIdx.x] += 1;
	}
}

void Evolver_gpu::frame_step_impl(f32 dt){
	internal_frame_step<<<CUDA_BLOCKS, CUDA_THREADS>>>(dt,
													   m_cuda_runner,
													   m_cuda_bullets,
													   m_cuda_tanks,
													   m_cuda_ai,
													   m_cuda_score);
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
	if(m_framecount % RETRIEVE_INTERVAL == 0){
		copy_from_device();
	}
}

void Evolver_gpu::evolve_ga_impl(){
	// copy over the data to the vector for easy sorting
	m_scoredata.resize(m_population_score.size());
	for(u32 i = 0; i < m_scoredata.size(); ++i){
		m_scoredata[i].first = i;
		m_scoredata[i].second = m_population_score[i];
	}
	
	// sort it from highest score to lowest score
	sort(m_scoredata.begin(), m_scoredata.end(), score_sort<u32>);
	if(m_last_score.size() == 0){
		m_last_score = m_scoredata;
	}else{
		bool result = equal(m_scoredata.begin(), m_scoredata.end(),
							m_last_score.begin());
		if(result){
			cout << "GENES DIDN'T EVOLVE" << endl;
		}
		m_last_score = m_scoredata;
	}
/*
	for(u32 i = 0; i < NUM_INSTANCES; ++i){
		cout << score_data[i].second << " ";
	}
	cout << endl;
*/
	// perform the reproduction process
	// score_data[n].first is the index to the individual
	// second is the score
	for(u32 i = 0; i < m_scoredata.size(); ++i){
		if(i < ELITE_COUNT){
			// copy over the elite genes
			copy_genes(&m_ai_b[i], &m_ai[m_scoredata[i].first]);
		}else{
			// time to reproduce given whatever else there may be
			// we'll force only the 1st half of the set of parents
			u32 p1 = rand() % m_scoredata.size()/2;
			u32 p2 = rand() % m_scoredata.size()/2;
			reproduce(&m_ai_b[i], &m_ai[m_scoredata[p1].first],
					  &m_ai[m_scoredata[p2].first]);

			// random chance to mutate
			u32 m = rand() % 100;
			if(m < MUTATION_RATE){
				mutate(&m_ai_b[i]);
			}
		}
	}
}

u32 Evolver_gpu::retrieve_score_impl(){
	u32 score = 0;
	u32 pos_found = 0;
	for(int i = 0; i < m_population_score.size(); ++i){
		if(m_population_score[i] > score){
			pos_found = i;
			score = m_population_score[i];
		}
	}
	return score;
}

void Evolver_gpu::save_best_gene_impl(const string& fname){
	// make sure that we have the stuff from the GPU
// 	copy_from_device();

	// find the individual with the highest score
// 	score_map::iterator best_pos;
// 	u32 score_find = retrieve_score_impl();
// 	for(best_pos = m_population_score.begin();
// 		best_pos != m_population_score.end(); ++best_pos){
// 		if(score_find == best_pos->second){
// 			break;
// 		}
// 	}
	
	ofstream fout(fname.c_str(), ios::trunc);
// 	fout.seekp(ios::end);

	// assume that we just want AI_CONTROLLER 0
	// write out the accel gene 1st
	if(fout.is_open()){
		u32 index = m_last_score[0].first;
		AI::AI_Core::gene_type tempval;
		for(int i = 0; i < MAX_GENE_DATA; ++i){
			tempval = m_ai[index].gene_accel[i][0];
			fout.write((const char*)&tempval, sizeof(tempval));
		}

		// write out the heading gene next
		for(int i = 0; i < MAX_GENE_DATA; ++i){
			tempval = m_ai[index].gene_heading[i][0];
			fout.write((const char*)&tempval, sizeof(tempval));
		}
	}
	
	fout.close();
}

void Evolver_gpu::prepare_game_state_impl(){
	m_population_score.clear();
	for(int i = 0; i < NUM_INSTANCES; ++i){
		m_population_score[i] = 0;
	}
	fill(m_score.begin(), m_score.end(), 0);
	m_framecount = 0;
	
	// get the backup buffer and put it into current one
// 	m_ai = m_ai_b;
// 	m_runner = m_runner_b;
// 	m_tanks = m_tanks_b;
// 	m_bullets = m_bullets_b;
	
	// setup stuff on the current buffer
// 	for(int i = 0; i < NUM_INSTANCES; ++i){
// 		Physics::vec2 params;
// 		params.x = -25;
// 		tank_id evading_tank = BasicTank::spawn_tank(&m_tanks[i],
// 													 params,
// 													 90,
// 													 0);
// 		AI::add_tank(&m_ai[i], evading_tank, AI_TYPE_EVADER);
// 		
// 		params.x = 3;
// 		params.y = 12;
// 		tank_id attacking_tank = BasicTank::spawn_tank(&m_tanks[i],
// 													   params,
// 													   180,
// 													   1);
// 		AI::add_tank(&m_ai[i], attacking_tank, AI_TYPE_ATTACKER);
// 		
// 		params.x = 20;
// 		params.y = -12;
// 		attacking_tank = BasicTank::spawn_tank(&m_tanks[i],
// 											   params,
// 											   180,
// 											   1);
// 		AI::add_tank(&m_ai[i], attacking_tank, AI_TYPE_ATTACKER);
// 	}
// 	
// 	// send it over to the device
// 	copy_to_device();
}


void Evolver_gpu::perpare_game_scenario_impl(u32 dist, u32 bullet_loc, u32 bullet_vec){
	m_framecount = 0;
	m_ai = m_ai_b;
	m_runner = m_runner_b;
	m_tanks = m_tanks_b;
	m_bullets = m_bullets_b;

	Physics::vec2 atk_params;

	// change the position based on the stuff above
	f32 theta = SECTOR_SIZE*bullet_loc + SECTOR_SIZE/2;
	f32 hypot = DISTANCE_FACTOR*dist + DISTANCE_FACTOR/2;
	atk_params.y = hypot * sin(util::degs_to_rads(theta));
	atk_params.x = hypot * cos(util::degs_to_rads(theta));
	f32 tank_rot = VECTOR_SIZE*bullet_vec + VECTOR_SIZE/2;

	for(int i = 0; i < NUM_INSTANCES; ++i){
		Physics::vec2 params;
		params.x = 0;
		tank_id evading_tank = BasicTank::spawn_tank(&m_tanks[i],
													 params,
													 90,
													 0);
		AI::add_tank(&m_ai[i], evading_tank, AI_TYPE_EVADER);

		tank_id attacking_tank = BasicTank::spawn_tank(&m_tanks[i],
													   atk_params,
													   tank_rot,
													   1);
		AI::add_tank(&m_ai[i], attacking_tank, AI_TYPE_ATTACKER);
	}

	// send it to the gpu
	copy_to_device();
}

void Evolver_gpu::end_game_scenario_impl(){
	copy_from_device();
	for(int i = 0; i < NUM_INSTANCES; ++i){
		if(m_tanks[i].state[0] != TANK_STATE_INACTIVE){
			if(m_population_score.find(i) == m_population_score.end()){
				m_population_score[i] = 0;
			}
			++(m_population_score[i]);
		}
	}
}

bool Evolver_gpu::is_game_over_impl(){
	if(m_framecount % RETRIEVE_INTERVAL == 0){
		bool all_done = true;
		bool really_done = true;
		for(int i = 0; i < NUM_INSTANCES; ++i){
			// tank 0 is the one dodging, check its status
			all_done &= (m_tanks[i].state[0] == TANK_STATE_INACTIVE);
			really_done &= (m_tanks[i].state[1] == TANK_STATE_INACTIVE);
		}
		if(all_done || really_done){
			// 			finalize_impl();
		}
		return all_done | really_done;
	}else{
		return false;
	}
}

void Evolver_gpu::finalize_impl(){
// 	for(int i = 0; i < NUM_INSTANCES; ++i){
// 		m_population_score[i] = m_score[i];
// 		m_score[i] = 0;
// 	}
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
	cudaMemcpy(m_cuda_score, &m_score[0],
			   NUM_INSTANCES*sizeof(u32), cudaMemcpyHostToDevice);
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
	cudaMemcpy(&m_score[0], m_cuda_score,
			   NUM_INSTANCES*sizeof(u32),
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
