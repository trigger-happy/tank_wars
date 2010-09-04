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
#include "game_core/tank_ai.h"

#define SHORT_MAX 32767
#define DISTANCE_DEFAULT SHORT_MAX

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
	aic->tc = tc;
	aic->bc = bc;
	aic->next_slot = 0;
	for(int i = 0; i < MAX_AI_CONTROLLERS; ++i){
		aic->controlled_tanks[i] = INVALID_ID;
	}
	memset(static_cast<void*>(aic->gene_accel),
		   0, MAX_AI_CONTROLLERS*MAX_GENE_DATA*sizeof(AI::AI_Core::gene_type));
	memset(static_cast<void*>(aic->gene_heading),
		   0, MAX_AI_CONTROLLERS*MAX_GENE_DATA*sizeof(AI::AI_Core::gene_type));
	
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
		// update registered tanks that are not invalid
		if(aic->controlled_tanks[idx] != INVALID_ID){
			update_perceptions(aic, idx);
			// use the info as an index to the genetic array
			// perform the action based values in the genetic array
			
			// future work here for some sort of state machine
		}
	}
}

void AI::add_tank(AI::AI_Core* aic, tank_id tid){
	if(aic->next_slot != MAX_AI_CONTROLLERS){
		aic->controlled_tanks[aic->next_slot] = tid;
		++(aic->next_slot);
	}
}

void AI::init_gene_data(AI::AI_Core* aic){
	srand(std::time(NULL));
	for(int i = 0; i < MAX_AI_CONTROLLERS; ++i){
		for(int j = 0; j < MAX_GENE_DATA; ++j){
			aic->gene_accel[j][i] = rand()%MAX_THRUST_VALUES;
			aic->gene_heading[j][i] = rand()%MAX_HEADING_VALUES;
		}
	}
}

void AI::update_perceptions(AI::AI_Core* aic,
							ai_id id){
	if(aic->controlled_tanks[id] != INVALID_ID){
		tank_id tid = aic->controlled_tanks[id];
		// reset the states, we'll use a single variable to store data
		// to optimize register usage
		s32 temp = -1;
		u32 dist = DISTANCE_DEFAULT;
		
		// get the nearest bullet
		bullet_id bid = AI::get_nearest_bullet(aic, tid);
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
		pos -= pos2;
		// save the result for later use
		pos2 = pos;
		pos.normalize();
		temp = AI::get_sector(aic, pos);
		aic->direction_state[id] = temp;
		
		// set the collision state
		f32 tspeed = Physics::PhysRunner::get_cur_velocity(rc, aic->tc->phys_id[tid]);
		dist = pos2.length();
		f32 bspeed = Physics::PhysRunner::get_cur_velocity(rc, aic->bc->phys_id[bid]);
		f32 tsp_adj = Physics::PhysRunner::get_velocity_vector(rc, aic->tc->phys_id[tid]) * pos;
		tsp_adj *= tspeed;
		f32 bsp_adj = Physics::PhysRunner::get_velocity_vector(rc, aic->bc->phys_id[bid]) * -pos;
		bsp_adj *= bspeed;
		tspeed = tsp_adj + bsp_adj;
		tspeed = min(tspeed, MAX_SPEED);
		temp = (s32)util::lerp(tspeed/MAX_SPEED, 0.0f, 9.0f);
	}
}

s32 AI::get_sector(AI::AI_Core* aic,
			   Physics::vec2 pos){
	f32 dir = atan2(pos.x, pos.y);
	dir = util::rads_to_degs(dir);
	dir = util::clamp_dir_360(dir);
	return ((s32)(dir/SECTOR_SIZE));
}