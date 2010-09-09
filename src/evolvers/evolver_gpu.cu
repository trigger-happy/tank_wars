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

void Evolver_gpu::evolve_ga_impl(){
}

u32 Evolver_gpu::retrieve_score_impl(){
	score_map::iterator best_pos;
	best_pos = max_element(m_population_score.begin(),
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
	fout.seekp(ios::end);

	// assume that we just want AI_CONTROLLER 0
	// write out the accel gene 1st
	u32 index = best_pos->first;
	for(int i = 0; i < MAX_GENE_DATA; ++i){
		fout << m_ai[index].gene_accel[i][0];
	}

	// write out the heading gene next
	for(int i = 0; i < MAX_GENE_DATA; ++i){
		fout << m_ai[index].gene_heading[i][0];
	}
	
	fout.close();
}

void Evolver_gpu::prepare_game_state_impl(){
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
		if(m_tanks[i].state[0] == TANK_STATE_INACTIVE){
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
