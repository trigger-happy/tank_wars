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
#include <cuda.h>
#include "exports.h"
#include "evolvers/evolver_gpu.h"
#include "../game_core/twgame.cu"

#define CUDA_BLOCKS		NUM_INSTANCES
#define CUDA_THREADS	MAX_ARRAY_SIZE

void Evolver_gpu::initialize_impl(){
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
}

void Evolver_gpu::retrieve_state_impl(){
	copy_from_device();
}

void Evolver_gpu::evolve_ga_impl(){
}

u32 Evolver_gpu::retrieve_score_impl(){
	return 0;
}

void Evolver_gpu::save_best_gene_impl(const std::string& fname){
}

void Evolver_gpu::prepare_game_state_impl(){
}

bool Evolver_gpu::is_game_over_impl(){
	return false;
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
				sizeof(AI::AI_Core), cudaMemcpyHostToDevice);
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
