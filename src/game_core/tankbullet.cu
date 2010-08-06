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
#include "game_core/tankbullet.h"
#include "game_core/physics.h"
#include "util/util.h"

#define OFFSCREEN_X 	-800
#define OFFSCREEN_Y 	800

#define MAX_BULLET_RANGE	 		30.0f
#define MAX_BULLET_VELOCITY 		20.0f
#define INITIAL_BULLET_ACCELERATION 1000.0f

using namespace Physics;

void TankBullet::initialize(PhysRunner* p){
	m_cur_free_bullet = 0;
	m_runner = p;
	vec2 params;
	//TODO: allocate all the bullet objects we need
	for(int i = 0; i < MAX_BULLETS; ++i){
		m_ids[i] = m_runner->create_object();
		// set the bullet data
		m_runner->set_rotation(i, 0);
		m_runner->set_shape_type(i, SHAPE_CIRCLE);
		params.x = 2; // y is ignored when shape is circle
		m_runner->set_dimensions(i, params);
		
		//TODO: change the 1 to ENTITY_BULLET
		m_runner->set_user_data(i, 1);
		
		// make it initially inactive
		deactivate(i);
	}
}

void TankBullet::destroy(){
	for(int i = 0; i < MAX_BULLETS; ++i){
		m_runner->destroy_object(m_ids[i]);
	}
}

void TankBullet::update(f32 dt){
	int idx = 0;
	#if __CUDA_ARCH__
		idx = threadIdx.x;
		if(idx < MAX_BULLETS){
	#elif !defined(__CUDA_ARCH__)
		for(idx = 0; idx < MAX_BULLETS; ++idx){
	#endif
	
			if(m_state[idx] == STATE_TRAVELLING){
				vec2 temp;
				temp = m_runner->get_cur_pos(m_ids[idx]);
				f32 xdiff = fabsf(m_initial_firing_pos.x[m_ids[idx]]
				- temp.x);
				f32 ydiff = fabsf(m_initial_firing_pos.y[m_ids[idx]]
				- temp.y);
				f32 sq_dist = (xdiff * xdiff) + (ydiff * ydiff);
				if(sq_dist >= MAX_BULLET_RANGE * MAX_BULLET_RANGE){
					deactivate(m_ids[idx]);
				}
			}
			
		}
}

void TankBullet::fire_bullet(bullet_id bid,
							 f32 rot_degrees,
							 vec2 pos){
	if(m_state[bid] != STATE_TRAVELLING){
		m_runner->set_rotation(bid, rot_degrees);
		m_initial_firing_pos.x[bid] = pos.x;
		m_initial_firing_pos.y[bid] = pos.y;
		vec2 params;
		f32 rotation_rads = util::degs_to_rads(rot_degrees);
		params.x = INITIAL_BULLET_ACCELERATION * cosf(rotation_rads);
		params.y = INITIAL_BULLET_ACCELERATION * sinf(rotation_rads);
		m_runner->set_acceleration(bid, params);
		
		m_runner->set_cur_pos(bid, pos);
		m_state[bid] = STATE_TRAVELLING;
	}
}

void TankBullet::deactivate(bullet_id bid){
	vec2 params;
	params.x = OFFSCREEN_X;
	params.y = OFFSCREEN_Y;
	m_runner->set_cur_pos(bid, params);
	
	m_runner->set_max_velocity(bid, MAX_BULLET_VELOCITY);
	
	params.x = 0;
	params.y = 0;
	m_runner->set_acceleration(bid, params);
	
	m_state[bid] = STATE_INACTIVE;
}

vec2 TankBullet::get_bullet_pos(bullet_id bid){
	return m_runner->get_cur_pos(m_ids[bid]);
}

bullet_id TankBullet::get_bullet(){
	return m_cur_free_bullet++;
}

bullet_id TankBullet::get_max_bullets() const{
	return MAX_BULLETS;
}