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
#include <memory.h>
#include "game_core/physics.h"
#include "util/util.h"

#define NUM_ITERATIONS 30

using namespace Physics;


vec2_array::vec2_array(){
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

vec2 vec2_array::get_vec2(u32 id){
	vec2 temp;
	temp.x = x[id];
	temp.y = y[id];
	return temp;
}


PhysRunner::PhysRunner()
: m_first_free_slot(0){
	memset(m_free_slots, 0, sizeof(u8)*MAX_ARRAY_SIZE);
}

PhysRunner::~PhysRunner(){
}

CUDA_EXPORT void update_verlet(f32 dt,
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

void PhysRunner::timestep(PhysRunner::RunnerCore* rc, f32 dt){
	// convert from millisecond to seconds
	dt /= 1000.0f;
	
	update_verlet(dt, &m_bodies);
}

void PhysRunner::find_next_free_slot(PhysRunner::RunnerCore* rc){
	// keep incrementing 
	for(u32 i = 0; i < MAX_ARRAY_SIZE; ++i){
		if(!m_free_slots[i]){
			m_first_free_slot = i;
			return;
		}
	}
	m_first_free_slot = MAX_ARRAY_SIZE;
}

u32 PhysRunner::get_slot(PhysRunner::RunnerCore* rc){
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

void PhysRunner::free_slot(PhysRunner::RunnerCore* rc, u32 id){
	m_free_slots[id] = false;
}


physBody::physBody(){
	for(int i = 0; i < MAX_ARRAY_SIZE; ++i){
		rotation[i] = 0;
		max_vel[i] = 0;
		can_collide[i] = false;
		shape_type[i] = 0;
	}
}

pBody PhysRunner::create_object(PhysRunner::RunnerCore* rc){
	return get_slot();
}

void PhysRunner::destroy_object(PhysRunner::RunnerCore* rc, pBody bd){
	free_slot(oid);
}

vec2 PhysRunner::get_cur_pos(PhysRunner::RunnerCore* rc, pBody bd){
	vec2 temp;
	temp.x = m_bodies.cur_pos.x[oid];
	temp.y = m_bodies.cur_pos.y[oid];
	return temp;
}

vec2 PhysRunner::get_acceleration(PhysRunner::RunnerCore* rc, pBody bd){
	vec2 temp;
	temp.x = m_bodies.acceleration.x[oid];
	temp.y = m_bodies.acceleration.y[oid];
	return temp;
}

f32 PhysRunner::get_rotation(PhysRunner::RunnerCore* rc, pBody bd){
	f32 rot = 0;
	rot = m_bodies.rotation[oid];
	return rot;
}

f32 PhysRunner::get_max_velocity(PhysRunner::RunnerCore* rc, pBody bd){
	f32 mv = 0;
	mv = m_bodies.rotation[oid];
	return mv;
}

bool PhysRunner::is_collidable(PhysRunner::RunnerCore* rc, pBody bd){
	bool f = false;
	f = m_bodies.can_collide[oid];
	return f;
}

pShape PhysRunner::get_shape_type(PhysRunner::RunnerCore* rc, pBody bd){
	u32 st = 0;
	st = m_bodies.shape_type[oid];
	return st;
}

u32 PhysRunner::get_user_data(PhysRunner::RunnerCore* rc, pBody bd){
	u32 ud = 0;
	ud = m_bodies.user_data[oid];
	return ud;
}

vec2 PhysRunner::get_dimensions(PhysRunner::RunnerCore* rc, pBody bd){
	vec2 dim;
	dim = m_bodies.dimension.get_vec2(oid);
	return dim;
}

void PhysRunner::set_cur_pos(pBody oid,
							 const Physics::vec2& pos){
	m_bodies.cur_pos.x[oid] = pos.x;
	m_bodies.cur_pos.y[oid] = pos.y;
	m_bodies.old_pos.x[oid] = pos.x;
	m_bodies.old_pos.y[oid] = pos.y;
}

void PhysRunner::set_acceleration(pBody oid,
								  const Physics::vec2& accel){
	m_bodies.acceleration.x[oid] = accel.x;
	m_bodies.acceleration.y[oid] = accel.y;
}

void PhysRunner::set_rotation(PhysRunner::RunnerCore* rc, pBody bd, f32 r){
	m_bodies.rotation[oid] = r;
}

void PhysRunner::set_max_velocity(PhysRunner::RunnerCore* rc, pBody bd, f32 mv){
	m_bodies.max_vel[oid] = mv;
}

void PhysRunner::should_collide(PhysRunner::RunnerCore* rc, pBody bd, bool f){
	m_bodies.can_collide[oid] = f;
}

void PhysRunner::set_shape_type(PhysRunner::RunnerCore* rc, pBody bd, pShape st){
	m_bodies.shape_type[oid] = st;
}

void PhysRunner::set_user_data(PhysRunner::RunnerCore* rc, pBody bd, u32 ud){
	m_bodies.user_data[oid] = ud;
}

void PhysRunner::set_dimensions(pBody oid,
								const Physics::vec2& dim){
	m_bodies.dimension.x[oid] = dim.x;
	m_bodies.dimension.y[oid] = dim.y;
}
