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

#define MAX_BULLET_RANGE	 		60.0f
#define MAX_BULLET_VELOCITY 		20.0f
#define INITIAL_BULLET_ACCELERATION 1000.0f

void TankBullet::initialize(TankBullet::BulletCollection* bc,
							Physics::PhysRunner::RunnerCore* p){
	bc->cur_free_bullet = 0;
	bc->parent_runner = p;
	Physics::vec2<s32> params;
	// allocate all the bullet objects we need
	for(int i = 0; i < MAX_BULLETS; ++i){
		bc->phys_id[i] = Physics::PhysRunner::create_object(p);
		// set the bullet data
		Physics::PhysRunner::set_rotation(p, i, 0);
		Physics::PhysRunner::set_shape_type(p, i, SHAPE_CIRCLE);
		params.x = BULLET_RADIUS; // y is ignored when shape is circle
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
	
		if(bc->state[idx] == BULLET_STATE_TRAVELLING){
				Physics::vec2<s32> temp, temp2;
				temp = Physics::PhysRunner::get_cur_pos(rc, bc->phys_id[idx]);
				temp2 = Physics::PhysRunner::get_prev_pos(rc, bc->phys_id[idx]);
				bc->travel_dist.x[bc->phys_id[idx]] += temp.x - temp2.x;
				bc->travel_dist.y[bc->phys_id[idx]] += temp.y - temp2.y;
				f32 dist = bc->travel_dist.get_vec2(idx).length();
				if(dist >= MAX_BULLET_RANGE){
					TankBullet::deactivate(bc, bc->phys_id[idx]);
				}
			}
			
		}
}

void TankBullet::fire_bullet(TankBullet::BulletCollection* bc,
							 bullet_id bid,
							 f32 rot_degrees,
							 Physics::vec2<s32> pos){
	if(bc->state[bid] != BULLET_STATE_TRAVELLING){
		Physics::PhysRunner::set_rotation(bc->parent_runner, bid, rot_degrees);
		bc->travel_dist.x[bid] = 0;
		bc->travel_dist.y[bid] = 0;
		Physics::vec2<f32> params;
		f32 rotation_rads = util::degs_to_rads(rot_degrees);
		params.x = INITIAL_BULLET_ACCELERATION * cosf(rotation_rads);
		params.y = INITIAL_BULLET_ACCELERATION * sinf(rotation_rads);
		Physics::PhysRunner::set_acceleration(bc->parent_runner,
											  bc->phys_id[bid], params);
		
		Physics::PhysRunner::set_cur_pos(bc->parent_runner,
										 bc->phys_id[bid], pos);
		bc->state[bid] = BULLET_STATE_TRAVELLING;
	}
}

void TankBullet::deactivate(TankBullet::BulletCollection* bc, bullet_id bid){
	Physics::PhysRunner::RunnerCore* rc = bc->parent_runner;
	Physics::vec2<s32> params;
	Physics::vec2<f32> paramsf;
	params.x = OFFSCREEN_X;
	params.y = OFFSCREEN_Y;
	Physics::PhysRunner::set_cur_pos(rc, bid, params);
	
	Physics::PhysRunner::set_max_velocity(rc, bid, MAX_BULLET_VELOCITY);
	
	params.x = 0;
	params.y = 0;
	Physics::PhysRunner::set_acceleration(rc, bid, paramsf);
	
	bc->state[bid] = TANK_STATE_INACTIVE;
}

Physics::vec2<s32> TankBullet::get_bullet_pos(TankBullet::BulletCollection* bc,
										 bullet_id bid){
	return Physics::PhysRunner::get_cur_pos(bc->parent_runner, bc->phys_id[bid]);
}

bullet_id TankBullet::get_bullet(TankBullet::BulletCollection* bc){
	return bc->cur_free_bullet++;
}

bullet_id TankBullet::get_max_bullets(TankBullet::BulletCollection* bc){
	return MAX_BULLETS;
}