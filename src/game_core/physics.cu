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
#include <cmath>
#include <algorithm>
#include <cassert>
#include "game_core/physics.h"
#include "util/util.h"

#define NUM_ITERATIONS 30

using namespace Physics;


__host__ __device__ vec2_array::vec2_array(){
	#if __CUDA_ARCH__
		// device code
		for(int i = 0; i < MAX_ARRAY_SIZE; ++i){
			x[i] = 0;
			y[i] = 0;
		}
	#elif !defined(__CUDA_ARCH__)
		// host code
		std::fill(x, x + MAX_ARRAY_SIZE, 0);
		std::fill(y, y + MAX_ARRAY_SIZE, 0);
	#endif
}

__host__ __device__ vec2 vec2_array::get_vec2(u32 id){
	vec2 temp;
	temp.x = x[id];
	temp.y = y[id];
	return temp;
}


__host__ PhysRunner::PhysRunner() 
: m_free_slots(0), m_first_free_slot(0)/*, m_update_dev_mem(false)*/{
}

__host__ PhysRunner::~PhysRunner(){
}

__host__ __device__ void update_verlet(f32 dt,
							  physBody* bodies){
	#if __CUDA_ARCH__
		// device code
		u32 idx = threadIdx.x;
		
		vec2 temp = bodies->cur_pos.get_vec2(idx);
		vec2 newpos;
		
		newpos.x = bodies->cur_pos.x[idx] - bodies->old_pos.x[idx] 
		+ bodies->acceleration.x[idx] * dt * dt;
		
		newpos.y = bodies->cur_pos.y[idx] - bodies->old_pos.y[idx] 
		+ bodies->acceleration.y[idx] * dt * dt;
		
		// don't let the object exceed maximum velocity
		vec2 maxvel;
		maxvel.x = fabsf(dt * bodies->max_vel[idx]
		* cosf(bodies->rotation[idx]
		* (static_cast<float>(PI)/180)));
		
		maxvel.y = fabsf(dt * bodies->max_vel[idx]
		* sinf(bodies->rotation[idx]
		* (static_cast<float>(PI)/180)));
		
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
	#elif !defined(__CUDA_ARCH__)
		// host code, this is completely serialized
		for(u32 idx = 0; idx < MAX_ARRAY_SIZE; ++idx){
			vec2 temp = bodies->cur_pos.get_vec2(idx);
			vec2 newpos;
			
			newpos.x = bodies->cur_pos.x[idx] - bodies->old_pos.x[idx] 
			+ bodies->acceleration.x[idx] * dt * dt;
			
			newpos.y = bodies->cur_pos.y[idx] - bodies->old_pos.y[idx] 
			+ bodies->acceleration.y[idx] * dt * dt;
			
			// don't let the object exceed maximum velocity
			vec2 maxvel;
			maxvel.x = fabsf(dt * bodies->max_vel[idx]
			* cosf(bodies->rotation[idx]
			* (static_cast<float>(PI)/180)));
			
			maxvel.y = fabsf(dt * bodies->max_vel[idx]
			* sinf(bodies->rotation[idx]
			* (static_cast<float>(PI)/180)));
			
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
	#endif
}

__host__ __device__ void PhysRunner::timestep(f32 dt){
	// convert from millisecond to seconds
	dt /= 1000.0f;
	
	update_verlet(dt, &m_bodies);
}

__host__ __device__ void PhysRunner::find_next_free_slot(){
	// keep incrementing 
	for(u32 i = 0; i < MAX_ARRAY_SIZE; ++i){
		if(!m_free_slots[i]){
			m_first_free_slot = i;
			return;
		}
	}
	m_first_free_slot = MAX_ARRAY_SIZE;
}

__host__ __device__ u32 PhysRunner::get_slot(){
	if(!m_free_slots[m_first_free_slot]){
		m_free_slots[m_first_free_slot] = true;
		return m_first_free_slot++;
	}
	
	find_next_free_slot();
	
	#if !defined(__CUDA_ARCH__)
		// host code
		assert(m_first_free_slot < MAX_ARRAY_SIZE);
	#endif
	
	//NOTE: be warned that we might be returning MAX_ARRAY_SIZE or greater
	return m_first_free_slot++;
}

__host__ __device__ void PhysRunner::free_slot(u32 id){
	m_free_slots[id] = false;
}


__host__ __device__ physBody::physBody(){
	for(int i = 0; i < MAX_ARRAY_SIZE; ++i){
		rotation[i] = 0;
		max_vel[i] = 0;
		can_collide[i] = false;
		shape_type[i] = 0;
	}
}

__host__ __device__ u32 PhysObject::create_object(PhysRunner* pr){
	return pr->get_slot();
}

__host__ __device__ void PhysObject::destroy_object(PhysRunner* pr, u32 oid){
	pr->free_slot(oid);
}

__host__ __device__ vec2 PhysObject::get_cur_pos(PhysRunner* pr, u32 oid){
	vec2 temp;
	temp.x = pr->m_bodies.cur_pos.x[oid];
	temp.y = pr->m_bodies.cur_pos.y[oid];
	return temp;
}

__host__ __device__ vec2 PhysObject::get_acceleration(PhysRunner* pr, u32 oid){
	vec2 temp;
	temp.x = pr->m_bodies.acceleration.x[oid];
	temp.y = pr->m_bodies.acceleration.y[oid];
	return temp;
}

__host__ __device__ f32 PhysObject::get_rotation(PhysRunner* pr, u32 oid){
	f32 rot = 0;
	rot = pr->m_bodies.rotation[oid];
	return rot;
}

__host__ __device__ f32 PhysObject::get_max_velocity(PhysRunner* pr, u32 oid){
	f32 mv = 0;
	mv = pr->m_bodies.rotation[oid];
	return mv;
}

__host__ __device__ bool PhysObject::is_collidable(PhysRunner* pr, u32 oid){
	bool f = false;
	f = pr->m_bodies.can_collide[oid];
	return f;
}

__host__ __device__ u32 PhysObject::get_shape_type(PhysRunner* pr, u32 oid){
	u32 st = 0;
	st = pr->m_bodies.shape_type[oid];
	return st;
}

__host__ __device__ u32 PhysObject::get_user_data(PhysRunner* pr, u32 oid){
	u32 ud = 0;
	ud = pr->m_bodies.user_data[oid];
	return ud;
}

__host__ __device__ vec2 PhysObject::get_dimensions(PhysRunner* pr, u32 oid){
	vec2 dim;
	dim = pr->m_bodies.dimension.get_vec2(oid);
	return dim;
}

__host__ __device__ void PhysObject::set_cur_pos(PhysRunner* pr, u32 oid,
												 const Physics::vec2& pos){
	pr->m_bodies.cur_pos.x[oid] = pos.x;
	pr->m_bodies.cur_pos.y[oid] = pos.y;
	pr->m_bodies.old_pos.x[oid] = pos.x;
	pr->m_bodies.old_pos.y[oid] = pos.y;
}

__host__ __device__ void PhysObject::set_acceleration(PhysRunner* pr, u32 oid,
													  const Physics::vec2& accel){
	pr->m_bodies.acceleration.x[oid] = accel.x;
	pr->m_bodies.acceleration.y[oid] = accel.y;
}

__host__ __device__ void PhysObject::set_rotation(PhysRunner* pr,
												  u32 oid, f32 r){
	pr->m_bodies.rotation[oid] = r;
}

__host__ __device__ void PhysObject::set_max_velocity(PhysRunner* pr,
													  u32 oid, f32 mv){
	pr->m_bodies.max_vel[oid] = mv;
}

__host__ __device__ void PhysObject::should_collide(PhysRunner* pr,
													u32 oid, bool f){
	pr->m_bodies.can_collide[oid] = f;
}

__host__ __device__ void PhysObject::set_shape_type(PhysRunner* pr,
													u32 oid, u32 st){
	pr->m_bodies.shape_type[oid] = st;
}

__host__ __device__ void PhysObject::set_user_data(PhysRunner* pr,
												   u32 oid, u32 ud){
	pr->m_bodies.user_data[oid] = ud;
}

__host__ __device__ void PhysObject::set_dimensions(PhysRunner* pr, u32 oid,
													const Physics::vec2& dim){
	pr->m_bodies.dimension.x[oid] = dim.x;
	pr->m_bodies.dimension.y[oid] = dim.y;
}
