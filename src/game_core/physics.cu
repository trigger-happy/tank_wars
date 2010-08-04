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

#include <algorithm>
#include <cassert>
#include "game_core/physics.h"
#include "util/util.h"

#define NUM_ITERATIONS 30

using namespace Physics;

#define VERLET_BLOCK 64
#define VERLET_GRID MAX_ARRAY_SIZE/VERLET_BLOCK

__host__ vec2_array::vec2_array(){
	std::fill(x, x + MAX_ARRAY_SIZE, 0);
	std::fill(y, y + MAX_ARRAY_SIZE, 0);
}

__host__ __device__ vec2 vec2_array::get_vec2(u32 id){
	vec2 temp;
	temp.x = x[id];
	temp.y = y[id];
	return temp;
}



__host__ physBody::physBody(){
	std::fill(rotation, rotation + MAX_ARRAY_SIZE, 0);
	std::fill(max_vel, max_vel + MAX_ARRAY_SIZE, 0);
	std::fill(can_collide, can_collide + MAX_ARRAY_SIZE, false);
}



__host__ __device__ PhysObject::PhysObject(PhysRunner* p) : m_runner(p){
	#if __CUDA_ARCH__ == 100
		//TODO: code here for creating a physobject from the device
	#elif !defined(__CUDA_ARCH__)
		// host code
		m_objid = m_runner->get_slot();
	#endif
}

__host__ __device__ PhysObject::~PhysObject(){
	#if __CUDA_ARCH__ == 100
		//TODO: code here for device path
	#elif !defined(__CUDA_ARCH__)
		// host code
		m_runner->free_slot(m_objid);
	#endif
}

__host__ __device__ vec2 PhysObject::get_cur_pos(){
	vec2 temp;
	
	#if __CUDA_ARCH__ == 100
		//TODO: code here for device path
	#elif !defined(__CUDA_ARCH__)
		// host path
		temp.x = m_runner->m_hostbodies.cur_pos.x[m_objid];
		temp.y = m_runner->m_hostbodies.cur_pos.y[m_objid];
	#endif
	
	return temp;
}

__host__ __device__ vec2 PhysObject::get_acceleration(){
	vec2 temp;
	
	#if __CUDA_ARCH__ == 100
		//TODO: code here for device path
	#elif !defined(__CUDA_ARCH__)
		// host path
		temp.x = m_runner->m_hostbodies.acceleration.x[m_objid];
		temp.y = m_runner->m_hostbodies.acceleration.y[m_objid];
	#endif
	
	return temp;
}

__host__ __device__ f32 PhysObject::get_rotation(){
	f32 rot = 0;
	#if __CUDA_ARCH__ == 100
		//TODO: code here for device path
	#elif !defined(__CUDA_ARCH__)
		// host path
		rot = m_runner->m_hostbodies.rotation[m_objid];
	#endif
	
	return rot;
}

__host__ __device__ f32 PhysObject::get_max_velocity(){
	f32 mv = 0;
	
	#if __CUDA_ARCH__ == 100
		//TODO: code here for device path
	#elif !defined(__CUDA_ARCH__)
		// host path
		mv = m_runner->m_hostbodies.rotation[m_objid];
	#endif
	
	return mv;
}

__host__ __device__ bool PhysObject::is_collidable(){
	bool f = false;
	
	#if __CUDA_ARCH__ == 100
		//TODO: code here for device path
	#elif !defined(__CUDA_ARCH__)
		// host path
		f = m_runner->m_hostbodies.can_collide[m_objid];
	#endif
	
	return f;
}

__host__ __device__ void PhysObject::set_cur_pos(const vec2& pos){
	#if __CUDA_ARCH__ == 100
		//TODO: code here for device path
	#elif !defined(__CUDA_ARCH__)
		// host path
		m_runner->m_hostbodies.cur_pos.x[m_objid] = pos.x;
		m_runner->m_hostbodies.cur_pos.y[m_objid] = pos.y;
		m_runner->m_hostbodies.old_pos.x[m_objid] = pos.x;
		m_runner->m_hostbodies.old_pos.y[m_objid] = pos.y;
		m_runner->update_dev_mem();
	#endif
}

__host__ __device__ void PhysObject::set_acceleration(const vec2& accel){
	#if __CUDA_ARCH__ == 100
		//TODO: code here for device path
	#elif !defined(__CUDA_ARCH__)
		// host path
		m_runner->m_hostbodies.acceleration.x[m_objid] = accel.x;
		m_runner->m_hostbodies.acceleration.y[m_objid] = accel.y;
		m_runner->update_dev_mem();
	#endif
}

__host__ __device__ void PhysObject::set_rotation(f32 r){
	#if __CUDA_ARCH__ == 100
		//TODO: code here for device path
	#elif !defined(__CUDA_ARCH__)
		// host path
		m_runner->m_hostbodies.rotation[m_objid] = r;
		m_runner->update_dev_mem();
	#endif
}

__host__ __device__ void PhysObject::set_max_velocity(f32 mv){
	#if __CUDA_ARCH__ == 100
		//TODO: code here for device path
	#elif !defined(__CUDA_ARCH__)
		// host path
		m_runner->m_hostbodies.max_vel[m_objid] = mv;
		m_runner->update_dev_mem();
	#endif
}

__host__ __device__ void PhysObject::should_collide(bool f){
	#if __CUDA_ARCH__ == 100
		//TODO: code here for device path
	#elif !defined(__CUDA_ARCH__)
		// host path
		m_runner->m_hostbodies.can_collide[m_objid] = f;
		m_runner->update_dev_mem();
	#endif
}



PhysRunner::PhysRunner() 
: m_free_slots(0), m_first_free_slot(0), m_update_dev_mem(false){
	
}

PhysRunner::~PhysRunner(){
	cudaFree(m_pdevbodies);
	cudaFree(m_pdevshapes);
}

__host__ void PhysRunner::initialize(){
	// initialize the device memory
	cudaMalloc(reinterpret_cast<void**>(&m_pdevbodies), sizeof(physBody));
	cudaMalloc(reinterpret_cast<void**>(&m_pdevshapes), sizeof(physShape));
	// copy over to the gpu
	copy_to_device();
}

__global__ void update_verlet(f32 dt, physBody* bodies, physShape* shapes){
	u32 idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	vec2 temp = bodies->cur_pos.get_vec2(idx);
	vec2 newpos;
	newpos.x = bodies->cur_pos.x[idx] - bodies->old_pos.x[idx] +
	bodies->acceleration.x[idx] * dt * dt;
	newpos.y = bodies->cur_pos.y[idx] - bodies->old_pos.y[idx] +
	bodies->acceleration.y[idx] * dt * dt;
	
	// don't let the object exceed maximum velocity
	vec2 maxvel;
	maxvel.x = fabsf(dt * bodies->max_vel[idx] * cosf(bodies->rotation[idx] *
	(static_cast<float>(PI)/180)));
	maxvel.y = fabsf(dt * bodies->max_vel[idx] * sinf(bodies->rotation[idx] *
	(static_cast<float>(PI)/180)));
	if(fabsf(newpos.x) > fabsf(maxvel.x)){
		if(bodies->acceleration.x[idx] < 0){
			newpos.x = -maxvel.x;
		}else{
			newpos.x = maxvel.x;
		}
	}
	if(fabsf(newpos.y) > fabsf(maxvel.y)){
		if(bodies->acceleration.y[idx] < 0){
			newpos.y = -maxvel.y;
		}else{
			newpos.y = maxvel.y;
		}
	}
	
	bodies->cur_pos.x[idx] += newpos.x;
	bodies->cur_pos.y[idx] += newpos.y;
	bodies->old_pos.x[idx] = temp.x;
	bodies->old_pos.y[idx] = temp.y;
}

__host__ void PhysRunner::timestep(f32 dt){
	if(m_update_dev_mem){
		copy_to_device();
		m_update_dev_mem = false;
	}
	
	// convert from millisecond to seconds
	dt /= 1000.0f;
	
	update_verlet<<<VERLET_BLOCK, VERLET_GRID>>>(dt, m_pdevbodies,
												 m_pdevshapes);
												 copy_from_device();
}

__host__ void PhysRunner::copy_from_device(){
	cudaMemcpy(&m_hostbodies, m_pdevbodies, sizeof(physBody),
			   cudaMemcpyDeviceToHost);
			   cudaMemcpy(&m_hostshapes, m_pdevshapes, sizeof(physShape),
						  cudaMemcpyDeviceToHost);
}

__host__ void PhysRunner::copy_to_device(){
	cudaMemcpy(m_pdevbodies, &m_hostbodies, sizeof(physBody),
			   cudaMemcpyHostToDevice);
			   cudaMemcpy(m_pdevshapes, &m_hostshapes, sizeof(physShape),
						  cudaMemcpyHostToDevice);
}

u32 PhysRunner::get_slot(){
	if(!m_free_slots[m_first_free_slot]){
		m_free_slots[m_first_free_slot] = true;
		return m_first_free_slot++;
	}
	find_next_free_slot();
	
	assert(m_first_free_slot < MAX_ARRAY_SIZE);
	
	return m_first_free_slot++;
}

void PhysRunner::free_slot(u32 id){
	m_free_slots[id] = false;
}

void PhysRunner::find_next_free_slot(){
	// keep incremenenting 
	for(u32 i = 0; i < MAX_ARRAY_SIZE; ++i){
		if(!m_free_slots[i]){
			m_first_free_slot = i;
			return;
		}
	}
	m_first_free_slot = MAX_ARRAY_SIZE;
}
