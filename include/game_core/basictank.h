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
#ifndef BASICTANK_H
#define BASICTANK_H
#include "exports.h"
#include "types.h"
#include "game_core/physics.h"
#include "game_core/tankbullet.h"

#define MAX_TANK_VEL	7.0f
#define INITIAL_ROT		0

#define TANK_ROT_RATE	2.0f
#define TANK_ACCEL_RATE	100.0f

#define TANK_WIDTH		4
#define TANK_LENGTH		6

#define STATE_INACTIVE			0
#define STATE_NEUTRAL			1
#define STATE_MOVING_FORWARD	2
#define STATE_MOVING_BACKWARD	3
#define STATE_FIRING			4
#define STATE_RELOADING			5

#define MAX_TANKS				MAX_ARRAY_SIZE/4
#define BULLETS_PER_TANK		1

typedef u32 tank_id;
typedef u32 tank_state;

namespace BasicTank{
	
	struct TankCollection{
		Physics::PhysRunner::RunnerCore* parent_runner;
		TankBullet::BulletCollection* bullet_collection;
		tank_state state[MAX_TANKS];
		u32	phys_id[MAX_TANKS];
		bullet_id bullet[MAX_TANKS][BULLETS_PER_TANK];
		bullet_id next_bullet[MAX_TANKS];
		tank_id next_tank;
	};
	
	/*!
	*/
	CUDA_HOST void initialize(TankCollection* tank,
							  Physics::PhysRunner::RunnerCore* p,
							  TankBullet::BulletCollection* bt);
							  
	/*!
	*/
	CUDA_HOST void reset_pointers(TankCollection* tank,
								  Physics::PhysRunner::RunnerCore* p,
								  TankBullet::BulletCollection* bt);
	
	/*!
	*/
	CUDA_HOST void destroy(TankCollection* tank);
	
	/*!
	*/
	CUDA_EXPORT void update(TankCollection* tt, f32 dt);
	
	/*!
	*/
	CUDA_EXPORT void move_forward(TankCollection* tt, tank_id tid);
	
	/*!
	*/
	CUDA_EXPORT void move_backward(TankCollection* tt, tank_id tid);
	
	/*!
	*/
	CUDA_EXPORT void stop(TankCollection* tt, tank_id tid);
	
	/*!
	*/
	CUDA_EXPORT void turn_left(TankCollection* tt, tank_id tid);
	
	/*!
	*/
	CUDA_EXPORT void turn_right(TankCollection* tt, tank_id tid);
	
	/*!
	*/
	CUDA_EXPORT void fire(TankCollection* tt, tank_id tid);
	
	/*!
	*/
	CUDA_EXPORT tank_id spawn_tank(TankCollection* tt,
								   const Physics::vec2& pos,
								   f32 rot);
	
	/*!
	*/
	CUDA_EXPORT void kill_tank(TankCollection* tt, tank_id tid);
	
	/*!
	*/
	Physics::vec2 get_tank_pos(TankCollection* tt, tank_id tid);
	
	/*!
	*/
	f32 get_tank_rot(TankCollection* tt, tank_id tid);
	
	/*!
	*/
	Physics::vec2 get_tank_accel(TankCollection* tt, tank_id tid);
};

#endif //BASICTANK_H