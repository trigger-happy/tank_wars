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
	// setup everything on the CPU
	for(u32 i = 0; i < NUM_INSTANCES; ++i){
		Physics::PhysRunner::initialize(&m_runner[i]);
		TankBullet::initialize(&m_bullets[i], &m_runner[i]);
		BasicTank::initialize(&m_tanks[i], &m_runner[i], &m_bullets[i]);
		AI::initialize(&m_ai[i], &m_tanks[i], &m_bullets[i]);
	}
	
	// now we start up the GPU stuff
	for(u32 i = 0; i < NUM_INSTANCES; ++i){
		cudaMalloc(reinterpret_cast<void**>(&m_cuda_runner[i]),
					sizeof(Physics::PhysRunner::RunnerCore));
		cudaMalloc(reinterpret_cast<void**>(&m_cuda_bullets[i]),
					sizeof(TankBullet::BulletCollection));
		cudaMalloc(reinterpret_cast<void**>(&m_cuda_tanks[i]),
					sizeof(BasicTank::TankCollection));
		cudaMalloc(reinterpret_cast<void**>(&m_cuda_ai[i]),
					sizeof(AI::AI_Core));
		
		// reset the pointers
		BasicTank::reset_pointers(&m_tanks[i], m_cuda_runner[i], m_cuda_bullets[i]);
		TankBullet::reset_phys_pointer(&m_bullets[i], m_cuda_runner[i]);
		
		// copy over the stuff to GPU mem
		cudaMemcpy(m_cuda_runner[i], &m_runner[i],
					sizeof(Physics::PhysRunner::RunnerCore), cudaMemcpyHostToDevice);
					
		cudaMemcpy(m_cuda_bullets[i], &m_bullets[i],
				   sizeof(TankBullet::BulletCollection), cudaMemcpyHostToDevice);
					
		cudaMemcpy(m_cuda_tanks[i], &m_tanks[i],
				   sizeof(BasicTank::TankCollection), cudaMemcpyHostToDevice);
					
		m_ai[i].bc = m_cuda_bullets[i];
		m_ai[i].tc = m_cuda_tanks[i];
		cudaMemcpy(m_cuda_ai[i], &m_ai[i],
				   sizeof(AI::AI_Core), cudaMemcpyHostToDevice);
	}
}

void Evolver_gpu::cleanup_impl(){
	for(u32 i = 0; i < NUM_INSTANCES; ++i){
		BasicTank::destroy(&m_tanks[i]);
		TankBullet::destroy(&m_bullets[i]);
		cudaFree(m_cuda_tanks[i]);
		cudaFree(m_cuda_bullets[i]);
		cudaFree(m_cuda_runner[i]);
		cudaFree(m_cuda_ai[i]);
	}
}

void Evolver_gpu::frame_step_impl(float dt){
}

void Evolver_gpu::retrieve_state_impl(){
}

void Evolver_gpu::evolve_ga_impl(){
}
