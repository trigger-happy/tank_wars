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

#define NUM_DISTANCE_STATES		4
#define DISTANCE_FACTOR			7.5

#define NUM_LOCATION_STATES		18
#define SECTOR_SIZE				20.0f

#define NUM_COLLISION_STATES	10
#define MAX_SPEED				20.0f //TODO: adjust this as needed

#define MAX_GENE_DATA			NUM_DISTANCE_STATES * NUM_LOCATION_STATES * NUM_COLLISION_STATES

#define MAX_THRUST_VALUES		3
#define MAX_HEADING_VALUES		18
#define MAX_AI_CONTROLLERS		MAX_TANKS

namespace AI{
	typedef u32 ai_id;
	// some AI specific info
	struct AI_Core{
		typedef u8 gene_type;
		BasicTank::TankCollection* tc;
		TankBullet::BulletCollection* bc;
		u32 next_slot;
		s32 collision_state[MAX_AI_CONTROLLERS];
		s32 direction_state[MAX_AI_CONTROLLERS];
		s32 distance_state[MAX_AI_CONTROLLERS];
		tank_id controlled_tanks[MAX_AI_CONTROLLERS];
		gene_type gene_accel[MAX_GENE_DATA][MAX_AI_CONTROLLERS];
		gene_type gene_heading[MAX_GENE_DATA][MAX_AI_CONTROLLERS];
	};
	
	// sensor functions
	/*!
	Get the nearest bullet to the tank tid (note, this gets enemy bullets)
	\param aic The AI_Core involved
	\param tid The tank id
	\return The bullet id of the closest bullet to the tank
	*/
	CUDA_EXPORT bullet_id get_nearest_bullet(AI_Core* aic,
											 tank_id tid);
												  
	/*!
	Get the nearest enemy tank
	\param aic The AI_Core involved
	\param tid The tank id
	\return The tank id of the closest enemy tank
	*/
	CUDA_EXPORT tank_id get_nearest_enemy(AI_Core* aic,
										  tank_id tid);
	
	/*!
	Get the nearest allied tank_id
	\param aic The AI_Core involved
	\param tid The tank id
	\return The tank id of the nearest allied tank
	*/
	CUDA_EXPORT tank_id get_nearest_ally(AI_Core* aic,
										 tank_id tid);
										 
	/*!
	Get the distance of a target tank to the current tank
	\param aic The AI_Core involved
	\param my_id The reference tank
	\param target_id The id of the target tank whose distance to check
	\return The distance between the 2 tanks
	*/
	CUDA_EXPORT f32 get_tank_dist(AI_Core* aic,
								  tank_id my_id,
								  tank_id target_id);
								 
	/*!
	Function for computing the sector where the bullet is.
	Also taken from the AI Game Engine programming book
	\param aic The AI_Core involved
	\param pos The position of the object
	*/
	CUDA_EXPORT s32 get_sector(AI_Core* aic,
							   Physics::vec2 pos);
								  
	/*!
	Update the perception data. Algorithm taken from the AI Game Engine
	Programming book.
	\param aic The AI_Core involved
	\param id The AI ID
	*/
	CUDA_EXPORT void update_perceptions(AI_Core* aic,
										ai_id id);
	
	/*!
	Get the distance of a bullet to the current tank
	\param aic The AI_Core involved
	\param tid The id of the reference tank
	\param The id of the bullet to query
	\return The distance between the bullet and the tank
	*/
	CUDA_EXPORT f32 get_bullet_dist(AI::AI_Core* aic,
									tank_id tid,
									bullet_id bid);
	
	/*!
	Initialize the AI_Core object
	*/
	CUDA_HOST void initialize(AI_Core* aic,
							  BasicTank::TankCollection* tc,
							  TankBullet::BulletCollection* bc);
	
	/*!
	Function called for each frame update
	\param aic The AI core
	\param dt The frame time in milliseconds
	*/
	CUDA_EXPORT void timestep(AI_Core* aic, f32 dt);
	
	/*!
	Add a new tank for the AI to control
	\param aic The AI core
	\param tid The tank id
	*/
	CUDA_EXPORT void add_tank(AI_Core* aic, tank_id tid);
	
	// genetic functions
	CUDA_HOST void init_gene_data(AI_Core* aic);
}

#endif //TANK_AI_H