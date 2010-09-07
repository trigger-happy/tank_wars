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

#define BULLET_STATE_INACTIVE		0
#define BULLET_STATE_TRAVELLING		1
#define BULLET_STATE_IMPACT			2

#define BULLET_RADIUS		0.2f

#define MAX_BULLETS			MAX_ARRAY_SIZE/2

typedef u32 bullet_id;

// structure of array
namespace TankBullet{
	
	struct BulletCollection{
		Physics::PhysRunner::RunnerCore* parent_runner;
		Physics::vec2_array travel_dist;
		u32 phys_id[MAX_BULLETS];
		u32 state[MAX_BULLETS];
		u32 faction[MAX_BULLETS];
		bullet_id cur_free_bullet;
	};
	
	/*!
	Initialize the tank bullets.
	\param p Pointer to a PhysRunner instance
	*/
	CUDA_HOST void initialize(BulletCollection* bc,
							  Physics::PhysRunner::RunnerCore* p);
	
	/*!
	*/
	CUDA_HOST void reset_phys_pointer(BulletCollection* bc,
									  Physics::PhysRunner::RunnerCore* p);
	
	/*!
	Destroy all tank bullets
	*/
	CUDA_HOST void destroy(BulletCollection* bc);
	
	/*!
	Perform a frame update for all tank bullets
	\param dt The delta time in seconds.
	*/
	CUDA_EXPORT void update(BulletCollection* bc, f32 dt);
	
	/*!
	Fire a bullet
	\param bid The bullet to fire
	\param rot_degrees The direction to fire the bullet
	\param pos The firing position of the bullet
	*/
	CUDA_EXPORT void fire_bullet(BulletCollection* bc,
								 bullet_id bid, f32 rot_degress,
								 Physics::vec2 pos);
					 
	/*!
	deactivate a bullet
	\param bid The bullet to deactivate
	*/
	CUDA_EXPORT void deactivate(BulletCollection* bc, bullet_id bid);
	
	/*!
	Get the current position of the bullet
	\param bid The bullet to query
	\return A vector containing the bullet's position
	*/
	CUDA_EXPORT Physics::vec2 get_bullet_pos(BulletCollection* bc,
											 bullet_id bid);
	
	/*!
	Get a bullet from the list
	\note DOES NOT CHECK IF IT'S GREATER THAN MAX_BULLETS
	*/
	CUDA_HOST bullet_id get_bullet(TankBullet::BulletCollection* bc);
	
	/*!
	Get the maximum number of bullets available
	*/
	CUDA_EXPORT bullet_id get_max_bullets(BulletCollection* bc);
};

#endif //TANKBULLET_H