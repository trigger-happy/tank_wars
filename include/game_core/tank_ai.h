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
	CUDA_EXPORT Physics::pBody get_nearest_bullet(BasicTank::TankCollection* tc,
												  tank_id tid);
												  
	CUDA_EXPORT Physics::pBody get_nearest_enemy(BasicTank::TankCollection* tc,
												 tank_id tid);
}

#endif //TANK_AI_H