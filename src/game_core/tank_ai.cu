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
#include <limits>
#include <cstring>
#include <ctime>
#include <iostream>
#include "exports.h"
#include "game_core/tank_ai.h"

#define SHORT_MAX 32767
#define DISTANCE_DEFAULT SHORT_MAX

// #define AI_PRINT_DBG

#ifdef AI_PRINT_DBG
#if !defined(__CUDA_ARCH__)
static int g_frame_count = 0;
#endif
#endif

using namespace std;

bullet_id AI::get_nearest_bullet(AI::AI_Core* aic,
								 tank_id tid){
	Physics::PhysRunner::RunnerCore* rc = aic->tc->parent_runner;
	u32 tank_faction = aic->tc->faction[tid];
	f32 sqdist = SHORT_MAX;
	unsigned int bid = INVALID_ID;
	for(int i = 0; i < MAX_BULLETS; ++i){
		// check if the current bullet is an enemy bullet
		if(aic->bc->faction[i] == tank_faction){
			// allied bullet, ignore
			continue;
		}
		
		// check if the bullet is active
		if(aic->bc->state[i] != BULLET_STATE_TRAVELLING){
			continue;
		}
		
		// get the distance
		f32 xdist = rc->bodies.cur_pos.x[aic->tc->phys_id[tid]]
		- rc->bodies.cur_pos.x[aic->bc->phys_id[i]];
		xdist *= xdist;
		f32 ydist = rc->bodies.cur_pos.y[aic->tc->phys_id[tid]]
		- rc->bodies.cur_pos.y[aic->bc->phys_id[i]];
		ydist *= ydist;
		
		// if the distance is smaller, save it
		if(xdist + ydist < sqdist){
			sqdist = xdist + ydist;
			bid = i;
		}
	}
	return bid;
}

tank_id AI::get_nearest_enemy(AI::AI_Core* aic,
							  tank_id tid){
	Physics::PhysRunner::RunnerCore* rc = aic->tc->parent_runner;
	u32 tank_faction = aic->tc->faction[tid];
	f32 sqdist = SHORT_MAX;
	unsigned int eid = INVALID_ID;
	for(int i = 0; i < MAX_TANKS; ++i){
		// check if the tank is an enemy tank
		if(aic->tc->faction[i] == tank_faction){
			// allied tank, ignore
			continue;
		}
		
		// check if the enemy is active
		if(aic->tc->state[i] == TANK_STATE_INACTIVE){
			continue;
		}
		// get the distance
		f32 xdist = rc->bodies.cur_pos.x[aic->tc->phys_id[tid]]
		- rc->bodies.cur_pos.x[aic->tc->phys_id[i]];
		xdist *= xdist;
		f32 ydist = rc->bodies.cur_pos.y[aic->tc->phys_id[tid]]
		- rc->bodies.cur_pos.y[aic->tc->phys_id[i]];
		ydist *= ydist;
		// if the distance is smaller, save it
		if(xdist + ydist < sqdist){
			sqdist = xdist + ydist;
			eid = i;
		}
	}
	return eid;
}

tank_id AI::get_nearest_ally(AI::AI_Core* aic,
							 tank_id tid){
	Physics::PhysRunner::RunnerCore* rc = aic->tc->parent_runner;
	u32 tank_faction = aic->tc->faction[tid];
	f32 sqdist = SHORT_MAX;
	unsigned int aid = INVALID_ID;
	for(int i = 0; i < MAX_TANKS; ++i){
		// check if the tank is an allied tank
		if(aic->tc->faction[i] != tank_faction){
			// enemy tank, ignore
			continue;
		}
		// get the distance
		f32 xdist = rc->bodies.cur_pos.x[aic->tc->phys_id[tid]]
		- rc->bodies.cur_pos.x[aic->tc->phys_id[i]];
		xdist *= xdist;
		f32 ydist = rc->bodies.cur_pos.y[aic->tc->phys_id[tid]]
		- rc->bodies.cur_pos.y[aic->tc->phys_id[i]];
		ydist *= ydist;
		// if the distance is smaller, save it
		if(xdist + ydist < sqdist){
			sqdist = xdist + ydist;
			aid = i;
		}
	}
	return aid;
}



f32 AI::get_tank_dist(AI::AI_Core* aic,
					  tank_id my_id,
					  tank_id target_id){
	Physics::PhysRunner::RunnerCore* rc = aic->tc->parent_runner;
	f32 xdist = rc->bodies.cur_pos.x[aic->tc->phys_id[my_id]]
	- rc->bodies.cur_pos.x[aic->tc->phys_id[target_id]];
	f32 ydist = rc->bodies.cur_pos.y[aic->tc->phys_id[my_id]]
	- rc->bodies.cur_pos.y[aic->tc->phys_id[target_id]];
	return sqrt((xdist*xdist) + (ydist*ydist));
}
								   
f32 AI::get_bullet_dist(AI::AI_Core* aic,
						tank_id tid,
						bullet_id bid){
	Physics::PhysRunner::RunnerCore* rc = aic->tc->parent_runner;
	f32 xdist = rc->bodies.cur_pos.x[aic->tc->phys_id[tid]]
	- rc->bodies.cur_pos.x[aic->bc->phys_id[bid]];
	f32 ydist = rc->bodies.cur_pos.y[aic->tc->phys_id[tid]]
	- rc->bodies.cur_pos.y[aic->bc->phys_id[bid]];
	return sqrt((xdist * xdist) + (ydist*ydist));
}

void AI::initialize(AI::AI_Core* aic,
					BasicTank::TankCollection* tc,
					TankBullet::BulletCollection* bc){
#ifdef AI_PRINT_DBG
#if !defined(__CUDA_ARCH__)
	g_frame_count = 0;
#endif
#endif
	aic->frame_count = FRAMES_PER_UPDATE-5;
	aic->tc = tc;
	aic->bc = bc;
	aic->next_slot = 0;
	for(int i = 0; i < MAX_AI_CONTROLLERS; ++i){
		aic->controlled_tanks[i] = INVALID_ID;
	}
	memset(static_cast<void*>(aic->gene_accel),
		   0, MAX_AI_EVADERS*MAX_GENE_DATA*sizeof(AI::AI_Core::gene_type));
	memset(static_cast<void*>(aic->gene_heading),
		   0, MAX_AI_EVADERS*MAX_GENE_DATA*sizeof(AI::AI_Core::gene_type));
	memset(static_cast<void*>(aic->bullet_vector),
		   0, MAX_AI_EVADERS*sizeof(s32));
	memset(static_cast<void*>(aic->tank_vector),
		   0, MAX_AI_EVADERS*sizeof(s32));
	memset(static_cast<void*>(aic->distance_state),
		   0, MAX_AI_EVADERS*sizeof(s32));
	memset(static_cast<void*>(aic->direction_state),
		   0, MAX_AI_EVADERS*sizeof(s32));
	memset(static_cast<void*>(aic->ai_type),
		   0, MAX_AI_CONTROLLERS*sizeof(s32));
	memset(static_cast<void*>(aic->shot_count),
		   0, MAX_AI_CONTROLLERS*sizeof(s32));

	for(int i = 0; i < MAX_AI_EVADERS; ++i){
		aic->desired_heading[i] = -1;
		aic->desired_thrust[i] = -1;
	}
	
	AI::init_gene_data(aic);
}

void AI::timestep(AI::AI_Core* aic, f32 dt){
	int idx = 0;
	#if __CUDA_ARCH__
	idx = threadIdx.x;
	if(idx < MAX_AI_CONTROLLERS){
	#elif !defined(__CUDA_ARCH__)
	for(idx = 0; idx < MAX_AI_CONTROLLERS; ++idx){
	#endif
		if(idx == 0){
			// convert from milliseconds to seconds
			dt /= 1000.0f;
			if(aic->frame_count >= FRAMES_PER_UPDATE){
				aic->frame_count = 0;
			}else{
				aic->frame_count += 1;
			}
		}
		#if __CUDA_ARCH__
// 		__syncthreads();
		#endif
		tank_id my_tank = aic->controlled_tanks[idx];
		// update registered tanks that are not invalid
		if(my_tank != INVALID_ID){
			if(aic->tc->state[my_tank] != TANK_STATE_INACTIVE){
				// DEBUG
				// 			BasicTank::move_forward(aic->tc, my_tank);
				// 			BasicTank::turn_left(aic->tc, my_tank);

				// update the sensors
				update_perceptions(aic, idx, dt);
#if __CUDA_ARCH__
// 				__syncthreads();
#endif

				// perform AI actions
				if(aic->ai_type[idx] == AI_TYPE_EVADER){
						// check how many frames has passed
						s32 index = 0;
					if(aic->frame_count >= FRAMES_PER_UPDATE){
						index = (aic->distance_state[idx] * NUM_LOCATION_STATES * NUM_BULLET_VECTORS)
								+ (aic->direction_state[idx] * NUM_BULLET_VECTORS)
								+ aic->bullet_vector[idx];
						
						if(index >= 0
							&& aic->bullet_vector[idx] >= 0
							&& aic->direction_state[idx] >= 0
							&& aic->distance_state[idx] >= 0){
							// valid index, access the action from the array
							
							// get the desired heading
							aic->desired_heading[idx] = aic->gene_heading[index][idx];
							aic->desired_thrust[idx] = aic->gene_accel[index][idx];

							// dump the data for debugging
							#ifdef AI_PRINT_DBG
							#if !defined(__CUDA_ARCH__)
							bullet_id near_bul = AI::get_nearest_bullet(aic, my_tank);
							Physics::vec2 mypos = BasicTank::get_tank_pos(aic->tc, my_tank);
							Physics::vec2 bpos = TankBullet::get_bullet_pos(aic->bc, near_bul);
							cout << g_frame_count << " "
									<< aic->bullet_vector[idx] << " "
									<< aic->tank_vector[idx] << " "
									<< aic->direction_state[idx] << " "
									<< aic->distance_state[idx] << " | "
									<< mypos.x << " "
									<< mypos.y << " "
									<< BasicTank::get_tank_rot(aic->tc, my_tank) << " | "
									<< bpos.x << " "
									<< bpos.y << " "
									<< Physics::PhysRunner::get_rotation(aic->bc->parent_runner, aic->bc->phys_id[near_bul])
									<< endl;
							#endif
							#endif
						}else{
							#if !defined(__CUDA_ARCH__)
// 							assert("NEGATIVE INDEX" && false);
							#endif
							BasicTank::stop(aic->tc, my_tank);
							aic->desired_heading[idx] = -1;
							aic->desired_thrust[idx] = -1;
							// this is for the desired_heading
							index = -1;
						}
						//TODO: state machine here in the future?
					}
					// perform the needed turn
					
					#if __CUDA_ARCH__
// 					__syncthreads();
					#endif
					// let's try to get to the right heading
					switch(aic->desired_thrust[idx]){
						case 0:
						case -1:
							BasicTank::stop(aic->tc, my_tank);
							break;
						case 1:
							BasicTank::move_forward(aic->tc, my_tank);
							break;
						case 2:
							BasicTank::move_backward(aic->tc, my_tank);
							break;
					}
					
					f32 cur_rot = Physics::PhysRunner::get_rotation(aic->tc->parent_runner,
																	aic->tc->phys_id[my_tank]);
					cur_rot = util::clamp_dir_360(cur_rot);
					cur_rot = AI::get_vector(cur_rot);
					if(aic->desired_heading[idx] != -1 && index != -1){
						if(aic->desired_heading[idx] < cur_rot){
							BasicTank::turn_left(aic->tc, my_tank);
						}else if(aic->desired_heading[idx] > cur_rot){
							BasicTank::turn_right(aic->tc, my_tank);
						}
					}
				}else if(aic->ai_type[idx] == AI_TYPE_ATTACKER){
					// get the nearest target to shoot at
					#if __CUDA_ARCH__
// 					__syncthreads();
					#endif
					tank_id tid = aic->controlled_tanks[idx];
					tank_id target = AI::get_nearest_enemy(aic, tid);
					if(target != INVALID_ID){
						
						// get the positions
						Physics::vec2 target_pos = BasicTank::get_tank_pos(aic->tc, target);
						Physics::vec2 my_pos = BasicTank::get_tank_pos(aic->tc, tid);
						target_pos -= my_pos;

						// get the necessary rotation
						f32 dir = atan2(target_pos.x, target_pos.y);
						dir = util::rads_to_degs(dir);
						dir = util::clamp_dir_360(dir);

						// cheat our rotation
						Physics::PhysRunner::set_rotation(aic->tc->parent_runner,
														aic->tc->phys_id[tid],
														90-dir);

						// fire away like a trigger-happy thing
// 						BasicTank::fire(aic->tc, tid);
						// copy over the trainer code:
						if(aic->shot_count[idx] == 0){
							BasicTank::fire(aic->tc, tid);
							aic->shot_count[idx] += 1;
						}else{
							bullet_id bid = aic->tc->bullet[tid][0];
							if(aic->bc->state[bid] == BULLET_STATE_INACTIVE){
								BasicTank::kill_tank(aic->tc, tid);
							}
						}
					}
				}else if(aic->ai_type[idx] == AI_TYPE_TRAINER){
					// just fire blindly
					if(aic->shot_count[idx] == 0){
						tank_id tid = aic->controlled_tanks[idx];
						BasicTank::fire(aic->tc, tid);
						aic->shot_count[idx] += 1;
					}else{
						tank_id tid = aic->controlled_tanks[idx];
						bullet_id bid = aic->tc->bullet[tid][0];
						if(aic->bc->state[bid] == BULLET_STATE_INACTIVE){
							BasicTank::kill_tank(aic->tc, tid);
						}
					}
				}
			}
		}
// 		#ifdef AI_PRINT_DBG
// 		#if !defined(__CUDA_ARCH__)
// 		if(g_frame_count == 0){
// 			// output the initial state
// 			tank_id my_tank = aic->controlled_tanks[idx];
// 			bullet_id near_bul = AI::get_nearest_bullet(aic, my_tank);
// 			Physics::vec2 mypos = BasicTank::get_tank_pos(aic->tc, my_tank);
// 			Physics::vec2 bpos = TankBullet::get_bullet_pos(aic->bc, near_bul);
// 			cout << g_frame_count << " "
// 					<< aic->bullet_vector[idx] << " "
// 					<< aic->tank_vector[idx] << " "
// 					<< aic->direction_state[idx] << " "
// 					<< aic->distance_state[idx] << " | "
// 					<< mypos.x << " "
// 					<< mypos.y << " "
// 					<< BasicTank::get_tank_rot(aic->tc, my_tank) << " | "
// 					<< bpos.x << " "
// 					<< bpos.y << " "
// 					<< Physics::PhysRunner::get_rotation(aic->bc->parent_runner, aic->bc->phys_id[near_bul])
// 					<< endl;
// 		}
// 		#endif
// 		#endif
	}
	#ifdef AI_PRINT_DBG
	#if !defined(__CUDA_ARCH__)
	++g_frame_count;
	#endif
	#endif
}

void AI::add_tank( AI::AI_Core* aic, tank_id tid, s32 ait){
	if(aic->next_slot != MAX_AI_CONTROLLERS){
		aic->controlled_tanks[aic->next_slot] = tid;
		aic->ai_type[aic->next_slot] = ait;
		++(aic->next_slot);
	}
}

void AI::init_gene_data(AI::AI_Core* aic){
	for(int i = 0; i < MAX_AI_EVADERS; ++i){
		for(int j = 0; j < MAX_GENE_DATA; ++j){
			aic->gene_accel[j][i] = static_cast<u8>(rand()%MAX_THRUST_VALUES);
			aic->gene_heading[j][i] = static_cast<u8>(rand()%MAX_HEADING_VALUES);
		}
	}
}

void AI::update_perceptions(AI::AI_Core* aic,
							ai_id id,
							f32 dt){
	//TODO: break this up into smaller functions
	if(aic->controlled_tanks[id] != INVALID_ID){
		tank_id tid = aic->controlled_tanks[id];
		if(aic->ai_type[id] == AI_TYPE_EVADER){
			// reset the states, we'll use a single variable to store data
			// to optimize register usage
			s32 temp = -1;
			u32 dist = DISTANCE_DEFAULT;

			// get the nearest bullet
			bullet_id bid = AI::get_nearest_bullet(aic, tid);
			if(bid == INVALID_ID){
				aic->distance_state[id] = -1;
				aic->direction_state[id] = -1;
				aic->tank_vector[id] = -1;
				aic->bullet_vector[id] = -1;
			}else{
				dist = AI::get_bullet_dist(aic, tid, bid);

				// set the distance state
				temp = min((u32)(dist/DISTANCE_FACTOR), NUM_DISTANCE_STATES);
				aic->distance_state[id] = temp;

				// set the direction state
				Physics::PhysRunner::RunnerCore* rc = aic->tc->parent_runner;
				Physics::vec2 pos = Physics::PhysRunner::get_cur_pos(rc,
																	aic->tc->phys_id[tid]);
				Physics::vec2 pos2 = Physics::PhysRunner::get_cur_pos(rc,
																	aic->bc->phys_id[bid]);
				pos2 -= pos;
				pos2.normalize();
				temp = AI::get_sector(pos2);
				aic->direction_state[id] = temp;

				// get the tank vector
				temp = Physics::PhysRunner::get_rotation(rc,
														aic->tc->phys_id[tid]);
				temp = AI::get_vector(temp);
				aic->tank_vector[id] = temp;

				// get the bullet vector
				temp = Physics::PhysRunner::get_rotation(rc,
														aic->bc->phys_id[bid]);
				temp = AI::get_vector(temp);
				aic->bullet_vector[id] = temp;
			}
		}else if(aic->ai_type[id] == AI_TYPE_ATTACKER){
		}
	}
}

s32 AI::get_sector(Physics::vec2 pos){
	f32 dir = atan2(pos.x, pos.y);
	dir = util::rads_to_degs(dir);
	dir = util::clamp_dir_360(90-dir);
	return ((s32)(dir/SECTOR_SIZE));
}

s32 AI::get_vector(f32 rot){
	rot = util::clamp_dir_360(rot);
	return ((s32)(rot/VECTOR_SIZE));
}
