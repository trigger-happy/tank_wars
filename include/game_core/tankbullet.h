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
#ifndef TANKBULLET_H
#define TANKBULLET_H
#include "types.h"
#include "game_core/physics.h"

#define STATE_INACTIVE		0
#define STATE_TRAVELLING	1
#define STATE_IMPACT		2

#define NUM_BULLETS			MAX_ARRAY_SIZE/2

typedef u32 bullet_id;

// structure of array
class TankBullet{
public:
	/*!
	Initialize the tank bullets.
	\param p Pointer to a PhysRunner instance
	*/
	CUDA_HOST void initialize(Physics::PhysRunner* p);
	
	/*!
	Destroy all tank bullets
	*/
	CUDA_HOST void destroy();
	
	/*!
	Perform a frame update for all tank bullets
	\param dt The delta time in seconds.
	*/
	CUDA_EXPORT void update(f32 dt);
	
	/*!
	Fire a bullet
	\param bid The bullet to fire
	\param rot_degrees The direction to fire the bullet
	\param pos The firing position of the bullet
	*/
	CUDA_EXPORT void fire_bullet(bullet_id bid, f32 rot_degress,
								 Physics::vec2 pos);
					 
	/*!
	deactivate a bullet
	\param bid The bullet to deactivate
	*/
	CUDA_EXPORT void deactivate(bullet_id bid);
	
	//TODO: add more stuff here
private:
	Physics::PhysRunner* m_runner;
	Physics::vec2_array m_initial_firing_pos;
	u32 m_ids[NUM_BULLETS];
	u32 m_state[NUM_BULLETS];
};

#endif //TANKBULLET_H