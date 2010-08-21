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
#ifndef TANK_AI_H
#define TANK_AI_H
#include "game_core/physics.h"
#include "game_core/basictank.h"
#include "game_core/collision_checker.h"
#include "util/util.h"

namespace AI{
	// sensor functions
	/*!
	Get the nearest bullet to the tank tid (note, this gets enemy bullets)
	\param tc The TankCollection which the tank belongs to
	\param bc The BulletCollection where the bullets reside
	\param tid The tank id
	\return The bullet id of the closest bullet to the tank
	*/
	CUDA_EXPORT bullet_id get_nearest_bullet(BasicTank::TankCollection* tc,
											 TankBullet::BulletCollection* bc,
											 tank_id tid);
												  
	/*!
	Get the nearest enemy tank
	\param tc The TankCollection which the tank belongs to
	\param tid The tank id
	\return The tank id of the closest enemy tank
	*/
	CUDA_EXPORT tank_id get_nearest_enemy(BasicTank::TankCollection* tc,
										  tank_id tid);
	
	/*!
	Get the nearest allied tank_id
	\param tc The TankCollection which the tank belongs to
	\param tid The tank id
	\return The tank id of the nearest allied tank
	*/
	CUDA_EXPORT tank_id get_nearest_ally(BasicTank::TankCollection* tc,
										 tank_id tid);
										 
	/*!
	Get the distance of a target tank to the current tank
	\param tc The TankCollection which the tank belongs to
	\param my_id The reference tank
	\param target_id The id of the target tank whose distance to check
	\return The distance between the 2 tanks
	*/
	CUDA_EXPORT uint32_t get_tank_dist(BasicTank::TankCollection* tc,
									   tank_id my_id,
									   tank_id target_id);
	
	/*!
	Get the distance of a bullet to the current tank
	\param tc The TankCollection which the tank belongs to
	\param bc The BulletCollection which the bullet belongs to
	\param tid The id of the reference tank
	\param The id of the bullet to query
	\return The distance between the bullet and the tank
	*/
	CUDA_EXPORT uint32_t get_bullet_dist(BasicTank::TankCollection* tc,
										 TankBullet::BulletCollection* bc,
										 tank_id tid,
										 bullet_id bid);
										 
}

#endif //TANK_AI_H