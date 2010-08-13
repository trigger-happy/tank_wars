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
#include "game_core/collision_checker.h"

bool Collision::bullet_tank_check(TankBullet::BulletCollection* bc,
								  BasicTank::TankCollection* tc,
								  bullet_id bid){
	//TODO: remove temporary vars to save on register space
	if(bc->parent_runner != tc->parent_runner){
		return false;
	}
	if(bc->state[bid] == STATE_INACTIVE){
		return false;
	}
	Physics::PhysRunner::RunnerCore* rc = bc->parent_runner;
	
	Physics::pBody bullet_pid = bc->phys_id[bid];
	f32 bullet_radius = rc->bodies.dimension.x[bullet_pid];
	for(unsigned int tid = 0; tid < MAX_TANKS; ++tid){
		//TODO: check if the tank is of the right faction
		if(bc->faction[bid] == tc->faction[tid]){
			// same faction, no friendly fire so ignore
			continue;
		}
		
		if(tc->state[tid] != STATE_INACTIVE){
			Physics::pBody tank_pid = tc->phys_id[tid];
			//TODO: change this to quad based check in the future
			f32 tank_radius = rc->bodies.dimension.x[tank_pid];
			Physics::vec2 tankpos, bulletpos;
			tankpos = Physics::PhysRunner::get_cur_pos(rc, tank_pid);
			bulletpos = Physics::PhysRunner::get_cur_pos(rc, bullet_pid);
			f32 xdiff = fabsf(tankpos.x - bulletpos.x);
			f32 ydiff = fabsf(tankpos.y - bulletpos.y);
			f32 sq_dist = (xdiff * xdiff) + (ydiff * ydiff);
			if(sq_dist <= (bullet_radius + tank_radius) * (bullet_radius + tank_radius)){
				// A hit!
				TankBullet::deactivate(bc, bid);
				BasicTank::kill_tank(tc, tid);
				return true;
			}
		}
	}
	return false;
}

bool Collision::tank_tank_check(BasicTank::TankCollection* tc,
								tank_id tid){
	if(tc->state[tid] == STATE_INACTIVE){
		return false;
	}
	Physics::PhysRunner::RunnerCore* rc = tc->parent_runner;
	Physics::pBody tank_pid = tc->phys_id[tid];
	
	//TODO: change this to quad measure in the future
	f32 t1_radius = rc->bodies.dimension.x[tank_pid];
	
	for(unsigned int tid2 = 0; tid2 < MAX_TANKS; ++tid2){
		if(tc->state[tid2] == STATE_INACTIVE){
			// not a living tank
			continue;
		}
		if(tc->faction[tid] == tc->faction[tid2]){
			// friendly tank
			continue;
		}
		
		// perform the actual check
		Physics::pBody tank2_pid = tc->phys_id[tid2];
		f32 t2_radius = rc->bodies.dimension.x[tank2_pid];
		//TODO: change this to quad based check in the future
		f32 tank_radius = rc->bodies.dimension.x[tank_pid];
		Physics::vec2 tank1pos, tank2pos;
		tank1pos = Physics::PhysRunner::get_cur_pos(rc, tank_pid);
		tank2pos = Physics::PhysRunner::get_cur_pos(rc, tank2_pid);
		f32 xdiff = fabsf(tank1pos.x - tank2pos.x);
		f32 ydiff = fabsf(tank1pos.y - tank2pos.y);
		f32 sq_dist = (xdiff * xdiff) + (ydiff * ydiff);
		if(sq_dist <= (t2_radius + tank_radius) * (t2_radius + tank_radius)){
			// A collision!
			BasicTank::kill_tank(tc, tid);
			BasicTank::kill_tank(tc, tid2);
			return true;
		}
	}
	return false;
}