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
#include "game_core/tank_ai.h"

#define SHORT_MAX 32767

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
	memset(static_cast<void*>(aic->genetic_data),
		   0, MAX_AI_CONTROLLERS*MAX_GENE_DATA*sizeof(int32_t));
}

void AI::timestep(AI::AI_Core* aic, f32 dt){
	int idx = 0;
	#if __CUDA_ARCH__
	idx = threadIdx.x;
	if(idx < MAX_AI_CONTROLLERS){
	#elif !defined(__CUDA_ARCH__)
	for(idx = 0; idx < MAX_AI_CONTROLLERS; ++idx){
	#endif
		// code here for performing AI update
		if(aic->controlled_tanks[idx] != INVALID_ID){
			BasicTank::move_forward(aic->tc, aic->controlled_tanks[idx]);
		}
	}
}

void AI::add_tank(AI::AI_Core* aic, tank_id tid){
	if(aic->next_slot != MAX_AI_CONTROLLERS){
		aic->controlled_tanks[aic->next_slot] = tid;
		++(aic->next_slot);
	}
}