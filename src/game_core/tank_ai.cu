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
#include "game_core/tank_ai.h"

bullet_id get_nearest_bullet(BasicTank::TankCollection* tc,
							 TankBullet::BulletCollection* bc,
							 tank_id tid){
	Physics::PhysRunner::RunnerCore* rc = tc->parent_runner;
	u32 tank_faction = tc->faction[tid];
	unsigned int sqdist = std::numeric_limits<unsigned int>::max();
	unsigned int bid = INVALID_ID;
	for(int i = 0; i < MAX_BULLETS; ++i){
		// check if the current bullet is an enemy bullet
		if(bc->faction[i] == tank_faction){
			// allied bullet, ignore
			continue;
		}
		
		// get the distance
		unsigned int xdist = rc->bodies.cur_pos.x[tc->phys_id[tid]]
		- rc->bodies.cur_pos.x[bc->phys_id[i]];
		xdist *= xdist;
		unsigned int ydist = rc->bodies.cur_pos.y[tc->phys_id[tid]]
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

tank_id get_nearest_enemy(BasicTank::TankCollection* tc,
						  tank_id tid){
}

tank_id get_nearest_ally(BasicTank::TankCollection* tc,
						 tank_id tid){
}