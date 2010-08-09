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
#include "game_core/tankbullet.h"
#include "game_core/physics.h"
#include "util/util.h"

#define OFFSCREEN_X 	-800
#define OFFSCREEN_Y 	800

#define MAX_BULLET_RANGE	 		30.0f
#define MAX_BULLET_VELOCITY 		20.0f
#define INITIAL_BULLET_ACCELERATION 1000.0f

void TankBullet::initialize(TankBullet::BulletCollection* bc,
							Physics::PhysRunner::RunnerCore* p){
	bc->cur_free_bullet = 0;
	bc->parent_runner = p;
	Physics::vec2 params;
	// allocate all the bullet objects we need
	for(int i = 0; i < MAX_BULLETS; ++i){
		bc->phys_id[i] = Physics::PhysRunner::create_object(p);
		// set the bullet data
		Physics::PhysRunner::set_rotation(p, i, 0);
		Physics::PhysRunner::set_shape_type(p, i, SHAPE_CIRCLE);
		params.x = 2; // y is ignored when shape is circle
		Physics::PhysRunner::set_dimensions(p, i, params);
		
		//TODO: change the 1 to ENTITY_BULLET
		Physics::PhysRunner::set_user_data(p, i, 1);
		
		// make it initially inactive
		TankBullet::deactivate(bc, i);
	}
}

void TankBullet::reset_phys_pointer(TankBullet::BulletCollection* bc,
									Physics::PhysRunner::RunnerCore* p){
	bc->parent_runner = p;
}

void TankBullet::destroy(TankBullet::BulletCollection* bc){
	for(int i = 0; i < MAX_BULLETS; ++i){
		Physics::PhysRunner::destroy_object(bc->parent_runner, bc->phys_id[i]);
	}
}

void TankBullet::update(TankBullet::BulletCollection* bc, f32 dt){
	Physics::PhysRunner::RunnerCore* rc = bc->parent_runner;
	int idx = 0;
	#if __CUDA_ARCH__
		idx = threadIdx.x;
		if(idx < MAX_BULLETS){
	#elif !defined(__CUDA_ARCH__)
		for(idx = 0; idx < MAX_BULLETS; ++idx){
	#endif
	
			if(bc->state[idx] == STATE_TRAVELLING){
				Physics::vec2 temp;
				temp = Physics::PhysRunner::get_cur_pos(rc, bc->phys_id[idx]);
				f32 xdiff = fabsf(bc->initial_pos.x[bc->phys_id[idx]]
				- temp.x);
				f32 ydiff = fabsf(bc->initial_pos.y[bc->phys_id[idx]]
				- temp.y);
				f32 sq_dist = (xdiff * xdiff) + (ydiff * ydiff);
				if(sq_dist >= MAX_BULLET_RANGE * MAX_BULLET_RANGE){
					TankBullet::deactivate(bc, bc->phys_id[idx]);
				}
			}
			
		}
}

void TankBullet::fire_bullet(TankBullet::BulletCollection* bc,
							 bullet_id bid,
							 f32 rot_degrees,
							 Physics::vec2 pos){
	if(bc->state[bid] != STATE_TRAVELLING){
		Physics::PhysRunner::set_rotation(bc->parent_runner, bid, rot_degrees);
		bc->initial_pos.x[bid] = pos.x;
		bc->initial_pos.y[bid] = pos.y;
		Physics::vec2 params;
		f32 rotation_rads = util::degs_to_rads(rot_degrees);
		params.x = INITIAL_BULLET_ACCELERATION * cosf(rotation_rads);
		params.y = INITIAL_BULLET_ACCELERATION * sinf(rotation_rads);
		Physics::PhysRunner::set_acceleration(bc->parent_runner,
											  bc->phys_id[bid], params);
		
		Physics::PhysRunner::set_cur_pos(bc->parent_runner,
										 bc->phys_id[bid], pos);
		bc->state[bid] = STATE_TRAVELLING;
	}
}

void TankBullet::deactivate(TankBullet::BulletCollection* bc, bullet_id bid){
	Physics::PhysRunner::RunnerCore* rc = bc->parent_runner;
	Physics::vec2 params;
	params.x = OFFSCREEN_X;
	params.y = OFFSCREEN_Y;
	Physics::PhysRunner::set_cur_pos(rc, bid, params);
	
	Physics::PhysRunner::set_max_velocity(rc, bid, MAX_BULLET_VELOCITY);
	
	params.x = 0;
	params.y = 0;
	Physics::PhysRunner::set_acceleration(rc, bid, params);
	
	bc->state[bid] = STATE_INACTIVE;
}

Physics::vec2 TankBullet::get_bullet_pos(TankBullet::BulletCollection* bc,
										 bullet_id bid){
	return Physics::PhysRunner::get_cur_pos(bc->parent_runner, bc->phys_id[bid]);
}

bullet_id TankBullet::get_bullet(TankBullet::BulletCollection* bc){
	return bc->cur_free_bullet++;
}

bullet_id TankBullet::get_max_bullets(TankBullet::BulletCollection* bc){
	return MAX_BULLETS;
}