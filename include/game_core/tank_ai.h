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
	Get the nearest bullet to the tank tid
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
}

#endif //TANK_AI_H