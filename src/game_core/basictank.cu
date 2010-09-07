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
#include "types.h"
#include "util/util.h"
#include "game_core/physics.h"
#include "game_core/basictank.h"

void BasicTank::initialize(BasicTank::TankCollection* tank,
						   Physics::PhysRunner::RunnerCore* p,
						   TankBullet::BulletCollection* tb){
	tank->parent_runner = p;
	tank->bullet_collection = tb;
	tank->next_tank = 0;
	Physics::vec2 params;
	for(int i = 0; i < MAX_TANKS; ++i){
		tank->phys_id[i] = Physics::PhysRunner::create_object(p);
		Physics::PhysRunner::set_acceleration(p, tank->phys_id[i], params);
		Physics::PhysRunner::set_max_velocity(p, tank->phys_id[i], MAX_TANK_VEL);
		Physics::PhysRunner::set_rotation(p, tank->phys_id[i], 0);
		//TODO: change this to shape_quad in the future
		Physics::PhysRunner::set_shape_type(p, tank->phys_id[i], SHAPE_CIRCLE);
		
// 		params.x = TANK_LENGTH;
// 		params.y = TANK_WIDTH;
// 		Physics::PhysRunner::set_dimensions(p, tank->phys_id[i], params);
		params.x = 2;
		Physics::PhysRunner::set_dimensions(p, tank->phys_id[i], params);
		
		//TODO: change this to ENTITY_TANK
		Physics::PhysRunner::set_user_data(p, tank->phys_id[i], 2);
		
		params.x = OFFSCREEN_X;
		params.y = OFFSCREEN_Y;
		Physics::PhysRunner::set_cur_pos(p, tank->phys_id[i], params);
		
		tank->next_bullet[i] = 0;
		tank->state[i] = TANK_STATE_INACTIVE;
		
		for(int j = 0; j < BULLETS_PER_TANK; ++j){
			if(j >= TankBullet::get_max_bullets(tb)){
				continue; // don't allocate anymore
			}
			tank->bullet[i][j] = TankBullet::get_bullet(tb);
		}
	}
}

void BasicTank::reset_pointers(BasicTank::TankCollection* tank,
							   Physics::PhysRunner::RunnerCore* p,
							   TankBullet::BulletCollection* bt){
	tank->parent_runner = p;
	tank->bullet_collection = bt;
}

void BasicTank::destroy(BasicTank::TankCollection* tank){
	for(int i = 0; i < MAX_TANKS; ++i){
		Physics::PhysRunner::destroy_object(tank->parent_runner,
											tank->phys_id[i]);
	}
}

void BasicTank::update(BasicTank::TankCollection* tt, f32 dt){
// 	int idx = 0;
// 	#if __CUDA_ARCH__
// 		idx = threadIdx.x;
// 		if(idx < MAX_TANKS){
// 	#elif !defined(__CUDA_ARCH__)
// 		for(idx = 0; idx < MAX_TANKS; ++idx){
// 	#endif
// 		}
}

void BasicTank::move_forward(BasicTank::TankCollection* tt, tank_id tid){
	Physics::vec2 accel;
	f32 rot = util::degs_to_rads(Physics::PhysRunner::get_rotation(tt->parent_runner,
																   tt->phys_id[tid]));
	accel.x = TANK_ACCEL_RATE * cosf(rot);
	accel.y = TANK_ACCEL_RATE * sinf(rot);
	Physics::PhysRunner::set_acceleration(tt->parent_runner,
										  tt->phys_id[tid], accel);
	tt->state[tid] = TANK_STATE_MOVING_FORWARD;
}

void BasicTank::move_backward(BasicTank::TankCollection* tt, tank_id tid){
	Physics::vec2 accel;
	f32 rot = util::degs_to_rads(Physics::PhysRunner::get_rotation(tt->parent_runner,
																   tt->phys_id[tid]));
	accel.x = -(TANK_ACCEL_RATE * cosf(rot));
	accel.y = -(TANK_ACCEL_RATE * sinf(rot));
	
	Physics::PhysRunner::set_acceleration(tt->parent_runner,
										  tt->phys_id[tid], accel);
	tt->state[tid] = TANK_STATE_MOVING_BACKWARD;
}

void BasicTank::stop(BasicTank::TankCollection* tt, tank_id tid){
	tt->state[tid] = TANK_STATE_NEUTRAL;
	Physics::vec2 accel;
	accel.x = accel.y = 0;
	Physics::PhysRunner::set_acceleration(tt->parent_runner,
										  tt->phys_id[tid], accel);
										  
	// re-use the vec2 object
	accel = Physics::PhysRunner::get_cur_pos(tt->parent_runner,
											 tt->phys_id[tid]);
	
	Physics::PhysRunner::set_cur_pos(tt->parent_runner,
									 tt->phys_id[tid],
									 accel);
}

void BasicTank::turn_left(BasicTank::TankCollection* tt, tank_id tid){
	f32 rot = Physics::PhysRunner::get_rotation(tt->parent_runner,
												tt->phys_id[tid]);
	
	rot += TANK_ROT_RATE;
	if(rot >= 360){
		rot -= 360;
	}
	
	Physics::PhysRunner::set_rotation(tt->parent_runner, tt->phys_id[tid], rot);
	if(tt->state[tid] == TANK_STATE_MOVING_FORWARD){
		BasicTank::move_forward(tt, tid);
	}else if(tt->state[tid] == TANK_STATE_MOVING_BACKWARD){
		BasicTank::move_backward(tt, tid);
	}
}

void BasicTank::turn_right(BasicTank::TankCollection* tt, tank_id tid){
	f32 rot = Physics::PhysRunner::get_rotation(tt->parent_runner,
												tt->phys_id[tid]);
	rot -= TANK_ROT_RATE;
	if(rot < 0){
		rot += 360;
	}
	
	Physics::PhysRunner::set_rotation(tt->parent_runner, tt->phys_id[tid], rot);
	if(tt->state[tid] == TANK_STATE_MOVING_FORWARD){
		BasicTank::move_forward(tt, tid);
	}else if(tt->state[tid] == TANK_STATE_MOVING_BACKWARD){
		BasicTank::move_backward(tt, tid);
	}
}

void BasicTank::fire(BasicTank::TankCollection* tt, tank_id tid){
	Physics::PhysRunner::RunnerCore* rc = tt->parent_runner;
	bullet_id nb = tt->next_bullet[tid]++;
	TankBullet::fire_bullet(tt->bullet_collection, tt->bullet[tid][nb],
							Physics::PhysRunner::get_rotation(rc, tt->phys_id[tid]),
							Physics::PhysRunner::get_cur_pos(rc, tt->phys_id[tid]));
	if(tt->next_bullet[tid] >= BULLETS_PER_TANK){
		tt->next_bullet[tid] = 0;
	}
}

tank_id BasicTank::spawn_tank(BasicTank::TankCollection* tt,
							  const Physics::vec2& pos,
							  f32 rot, u32 faction){
	tank_id tid = tt->next_tank++;
	Physics::PhysRunner::set_cur_pos(tt->parent_runner, tt->phys_id[tid], pos);
	Physics::PhysRunner::set_rotation(tt->parent_runner, tt->phys_id[tid], rot);
	Physics::vec2 accel;
	accel.x = accel.y = 0;
	Physics::PhysRunner::set_acceleration(tt->parent_runner, tt->phys_id[tid], accel);
	tt->state[tid] = TANK_STATE_NEUTRAL;
	tt->faction[tid] = faction;
	
	for(int i = 0; i < BULLETS_PER_TANK; ++i){
		tt->bullet_collection->faction[tt->bullet[tid][i]] = faction;
	}
	return tid;
}

void BasicTank::kill_tank(BasicTank::TankCollection* tt, tank_id tid){
	tt->state[tid] = TANK_STATE_INACTIVE;
	Physics::PhysRunner::RunnerCore* rc = tt->parent_runner;
	Physics::vec2 params;
	params.x = OFFSCREEN_X;
	params.y = OFFSCREEN_Y;
	Physics::PhysRunner::set_cur_pos(rc, tt->phys_id[tid], params);
	
	params.x = 0;
	params.y = 0;
	Physics::PhysRunner::set_acceleration(rc, tt->phys_id[tid], params);
}

Physics::vec2 BasicTank::get_tank_pos(BasicTank::TankCollection* tt, tank_id tid){
	return Physics::PhysRunner::get_cur_pos(tt->parent_runner, tt->phys_id[tid]);
}

f32 BasicTank::get_tank_rot(BasicTank::TankCollection* tt, tank_id tid){
	return Physics::PhysRunner::get_rotation(tt->parent_runner, tt->phys_id[tid]);
}

Physics::vec2 BasicTank::get_tank_accel(BasicTank::TankCollection* tt,
										tank_id tid){
	return Physics::PhysRunner::get_acceleration(tt->parent_runner, tt->phys_id[tid]);
}