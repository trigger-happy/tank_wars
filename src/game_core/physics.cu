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
#include "game_core/physics.h"

Physics::vec2_array::vec2_array(){
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

Physics::vec2 Physics::vec2_array::get_vec2(u32 id){
	vec2 temp;
	temp.x = x[id];
	temp.y = y[id];
	return temp;
}


void Physics::PhysRunner::initialize(Physics::PhysRunner::RunnerCore* rc){
	memset(rc->free_slots, 0, sizeof(u8)*MAX_ARRAY_SIZE);
}

void Physics::PhysRunner::cleanup(Physics::PhysRunner::RunnerCore* rc){
	//TODO: cleanup stuff here
}

CUDA_EXPORT void update_verlet(f32 dt,
							   Physics::physBody* bodies){
	int idx = 0;
	#if __CUDA_ARCH__
		// device code
	idx = threadIdx.x;

	__syncthreads();
	
	#elif !defined(__CUDA_ARCH__)
		// host code, this is completely serialized
	for(idx = 0; idx < MAX_ARRAY_SIZE; ++idx){
	#endif
	Physics::vec2 temp = bodies->cur_pos.get_vec2(idx);
	Physics::vec2 newpos;
	
	newpos.x = bodies->cur_pos.x[idx] - bodies->old_pos.x[idx] 
	+ bodies->acceleration.x[idx] * dt * dt;
	
	newpos.y = bodies->cur_pos.y[idx] - bodies->old_pos.y[idx] 
	+ bodies->acceleration.y[idx] * dt * dt;
	
	// don't let the object exceed maximum velocity
	Physics::vec2 maxvel;
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
	
	#if !defined(__CUDA_ARCH__)
	}
	#endif
}

void Physics::PhysRunner::timestep(Physics::PhysRunner::RunnerCore* rc, f32 dt){
	// convert from millisecond to seconds
	dt /= 1000.0f;
	
	update_verlet(dt, &(rc->bodies));
}

void Physics::PhysRunner::find_next_free_slot(Physics::PhysRunner::RunnerCore* rc){
	// keep incrementing 
	for(u32 i = 0; i < MAX_ARRAY_SIZE; ++i){
		if(!rc->free_slots[i]){
			rc->first_free_slot = i;
			return;
		}
	}
	rc->first_free_slot= MAX_ARRAY_SIZE;
}

u32 Physics::PhysRunner::get_slot(Physics::PhysRunner::RunnerCore* rc){
	if(!rc->free_slots[rc->first_free_slot]){
		rc->free_slots[rc->first_free_slot] = true;
		return rc->first_free_slot++;
	}
	
	Physics::PhysRunner::find_next_free_slot(rc);
	
	#if !defined(__CUDA_ARCH__)
		// host code
		assert(rc->first_free_slot < MAX_ARRAY_SIZE);
	#endif
	
	//WARNING: we might be returning MAX_ARRAY_SIZE or greater
	return rc->first_free_slot++;
}

void Physics::PhysRunner::free_slot(Physics::PhysRunner::RunnerCore* rc, u32 id){
	rc->free_slots[id] = false;
}


void Physics::init_physbody(Physics::physBody* pb){
	int idx = 0;
	#if __CUDA_ARCH__
	idx = threadIdx.x;
	if(idx < MAX_ARRAY_SIZE){
	#elif !defined(__CUDA_ARCH__)
	for(idx = 0; idx < MAX_ARRAY_SIZE; ++idx){
	#endif
		pb->rotation[idx] = 0;
		pb->max_vel[idx] = 0;
		pb->can_collide[idx] = false;
		pb->shape_type[idx] = 0;
	}
}

Physics::pBody Physics::PhysRunner::create_object(Physics::PhysRunner::RunnerCore* rc){
	return Physics::PhysRunner::get_slot(rc);
}

void Physics::PhysRunner::destroy_object(Physics::PhysRunner::RunnerCore* rc,
										 Physics::pBody oid){
	Physics::PhysRunner::free_slot(rc, oid);
}

Physics::vec2 Physics::PhysRunner::get_cur_pos(Physics::PhysRunner::RunnerCore* rc,
											   Physics::pBody oid){
	vec2 temp;
	temp.x = rc->bodies.cur_pos.x[oid];
	temp.y = rc->bodies.cur_pos.y[oid];
	return temp;
}

Physics::vec2 Physics::PhysRunner::get_acceleration(Physics::PhysRunner::RunnerCore* rc,
													Physics::pBody oid){
	vec2 temp;
	temp.x = rc->bodies.acceleration.x[oid];
	temp.y = rc->bodies.acceleration.y[oid];
	return temp;
}

f32 Physics::PhysRunner::get_rotation(Physics::PhysRunner::RunnerCore* rc,
									  Physics::pBody oid){
	f32 rot = 0;
	rot = rc->bodies.rotation[oid];
	return rot;
}

f32 Physics::PhysRunner::get_max_velocity(Physics::PhysRunner::RunnerCore* rc,
										  Physics::pBody oid){
	f32 mv = 0;
	mv = rc->bodies.rotation[oid];
	return mv;
}

bool Physics::PhysRunner::is_collidable(Physics::PhysRunner::RunnerCore* rc,
										Physics::pBody oid){
	bool f = false;
	f = rc->bodies.can_collide[oid];
	return f;
}

Physics::pShape Physics::PhysRunner::get_shape_type(Physics::PhysRunner::RunnerCore* rc,
													Physics::pBody oid){
	u32 st = 0;
	st = rc->bodies.shape_type[oid];
	return st;
}

u32 Physics::PhysRunner::get_user_data(Physics::PhysRunner::RunnerCore* rc,
									   Physics::pBody oid){
	u32 ud = 0;
	ud = rc->bodies.user_data[oid];
	return ud;
}

Physics::vec2 Physics::PhysRunner::get_dimensions(Physics::PhysRunner::RunnerCore* rc,
												  Physics::pBody oid){
	vec2 dim;
	dim = rc->bodies.dimension.get_vec2(oid);
	return dim;
}

void Physics::PhysRunner::set_cur_pos(Physics::PhysRunner::RunnerCore* rc,
									  Physics::pBody oid,
									  const Physics::vec2& pos){
	rc->bodies.cur_pos.x[oid] = pos.x;
	rc->bodies.cur_pos.y[oid] = pos.y;
	rc->bodies.old_pos.x[oid] = pos.x;
	rc->bodies.old_pos.y[oid] = pos.y;
}

void Physics::PhysRunner::set_acceleration(Physics::PhysRunner::RunnerCore* rc,
										   Physics::pBody oid,
										   const Physics::vec2& accel){
	rc->bodies.acceleration.x[oid] = accel.x;
	rc->bodies.acceleration.y[oid] = accel.y;
}

void Physics::PhysRunner::set_rotation(Physics::PhysRunner::RunnerCore* rc,
									   Physics::pBody oid, f32 r){
	rc->bodies.rotation[oid] = r;
}

void Physics::PhysRunner::set_max_velocity(Physics::PhysRunner::RunnerCore* rc,
										   Physics::pBody oid, f32 mv){
	rc->bodies.max_vel[oid] = mv;
}

void Physics::PhysRunner::should_collide(Physics::PhysRunner::RunnerCore* rc,
										 Physics::pBody oid, bool f){
	rc->bodies.can_collide[oid] = f;
}

void Physics::PhysRunner::set_shape_type(Physics::PhysRunner::RunnerCore* rc,
										 Physics::pBody oid, pShape st){
	rc->bodies.shape_type[oid] = st;
}

void Physics::PhysRunner::set_user_data(Physics::PhysRunner::RunnerCore* rc,
										Physics::pBody oid, u32 ud){
	rc->bodies.user_data[oid] = ud;
}

void Physics::PhysRunner::set_dimensions(Physics::PhysRunner::RunnerCore* rc,
										 Physics::pBody oid,
										 const Physics::vec2& dim){
	rc->bodies.dimension.x[oid] = dim.x;
	rc->bodies.dimension.y[oid] = dim.y;
}
