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
#define OFFSCREEN_X		1000
#define OFFSCREEN_Y		1000
#define INITIAL_ROT		0

#define TANK_ROT_RATE	2.0f
#define TANK_ACCEL_RATE	100.0f

#define TANK_WIDTH		2
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

class BasicTank{
public:
	/*!
	*/
	CUDA_HOST void initialize(Physics::PhysRunner* p,
							  TankBullet* tb);
	
	/*!
	*/
	CUDA_HOST void destroy();
	
	/*!
	*/
	CUDA_EXPORT void update(f32 dt);
	
	/*!
	*/
	CUDA_EXPORT void move_forward(tank_id tid);
	
	/*!
	*/
	CUDA_EXPORT void move_backward(tank_id tid);
	
	/*!
	*/
	CUDA_EXPORT void stop(tank_id tid);
	
	/*!
	*/
	CUDA_EXPORT void turn_left(tank_id tid);
	
	/*!
	*/
	CUDA_EXPORT void turn_right(tank_id tid);
	
	/*!
	*/
	CUDA_EXPORT void fire(tank_id tid);
	
private:
	Physics::PhysRunner* m_runner;
	TankBullet* m_tb;
	tank_state m_state[MAX_TANKS];
	u32	m_ids[MAX_TANKS];
	bullet_id m_bullet[MAX_TANKS][BULLETS_PER_TANK];
	bullet_id next_bullet[MAX_TANKS];
};

#endif //BASICTANK_H