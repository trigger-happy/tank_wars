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
#include <boost/integer_traits.hpp>
#include "game_core/tank_ai.h"

bullet_id AI::get_nearest_bullet(BasicTank::TankCollection* tc,
								 TankBullet::BulletCollection* bc,
								 tank_id tid){
	Physics::PhysRunner::RunnerCore* rc = tc->parent_runner;
	u32 tank_faction = tc->faction[tid];
	unsigned int sqdist = boost::integer_traits<unsigned int>::const_max;
	unsigned int bid = INVALID_ID;
	for(int i = 0; i < MAX_BULLETS; ++i){
		// check if the current bullet is an enemy bullet
		if(bc->faction[i] == tank_faction){
			// allied bullet, ignore
			continue;
		}
		
		// get the distance
		int xdist = rc->bodies.cur_pos.x[tc->phys_id[tid]]
		- rc->bodies.cur_pos.x[bc->phys_id[i]];
		xdist *= xdist;
		int ydist = rc->bodies.cur_pos.y[tc->phys_id[tid]]
		- rc->bodies.cur_pos.y[bc->phys_id[i]];
		ydist *= ydist;
		
		// if the distance is smaller, save it
		if(xdist + ydist < sqdist){
			sqdist = xdist + ydist;
			bid = i;
		}
	}
	return bid;
}

tank_id AI::get_nearest_enemy(BasicTank::TankCollection* tc,
							  tank_id tid){
	Physics::PhysRunner::RunnerCore* rc = tc->parent_runner;
	u32 tank_faction = tc->faction[tid];
	unsigned int sqdist = boost::integer_traits<unsigned int>::const_max;
	unsigned int eid = INVALID_ID;
	for(int i = 0; i < MAX_TANKS; ++i){
		// check if the tank is an enemy tank
		if(tc->faction[i] == tank_faction){
			// allied tank, ignore
			continue;
		}
		// get the distance
		int xdist = rc->bodies.cur_pos.x[tc->phys_id[tid]]
		- rc->bodies.cur_pos.x[tc->phys_id[i]];
		xdist *= xdist;
		int ydist = rc->bodies.cur_pos.y[tc->phys_id[tid]]
		- rc->bodies.cur_pos.y[tc->phys_id[i]];
		ydist *= ydist;
		// if the distance is smaller, save it
		if(xdist + ydist < sqdist){
			sqdist = xdist + ydist;
			eid = i;
		}
	}
	return eid;
}

tank_id AI::get_nearest_ally(BasicTank::TankCollection* tc,
							 tank_id tid){
	Physics::PhysRunner::RunnerCore* rc = tc->parent_runner;
	u32 tank_faction = tc->faction[tid];
	unsigned int sqdist = boost::integer_traits<unsigned int>::const_max;
	unsigned int aid = INVALID_ID;
	for(int i = 0; i < MAX_TANKS; ++i){
		// check if the tank is an allied tank
		if(tc->faction[i] != tank_faction){
			// enemy tank, ignore
			continue;
		}
		// get the distance
		int xdist = rc->bodies.cur_pos.x[tc->phys_id[tid]]
		- rc->bodies.cur_pos.x[tc->phys_id[i]];
		xdist *= xdist;
		int ydist = rc->bodies.cur_pos.y[tc->phys_id[tid]]
		- rc->bodies.cur_pos.y[tc->phys_id[i]];
		ydist *= ydist;
		// if the distance is smaller, save it
		if(xdist + ydist < sqdist){
			sqdist = xdist + ydist;
			aid = i;
		}
	}
	return aid;
}



CUDA_EXPORT uint32_t AI::get_tank_dist(BasicTank::TankCollection* tc,
									   tank_id my_id,
									   tank_id target_id){
}
								   
CUDA_EXPORT uint32_t AI::get_bullet_dist(BasicTank::TankCollection* tc,
										 TankBullet::BulletCollection* bc,
										 tank_id tid,
										 bullet_id bid){
}