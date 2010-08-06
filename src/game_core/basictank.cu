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
#include "types.h"
#include "util/util.h"
#include "game_core/physics.h"
#include "game_core/basictank.h"

using namespace Physics;

void BasicTank::initialize(PhysRunner* p,
						   TankBullet* tb){
	m_runner = p;
	m_tb = tb;
	m_next_tank = 0;
	vec2 params;
	for(int i = 0; i < MAX_TANKS; ++i){
		m_ids[i] = m_runner->create_object();
		m_runner->set_acceleration(m_ids[i], params);
		m_runner->set_max_velocity(m_ids[i], MAX_TANK_VEL);
		m_runner->set_rotation(m_ids[i], 0);
		m_runner->set_shape_type(m_ids[i], SHAPE_QUAD);
		
		params.x = TANK_LENGTH;
		params.y = TANK_WIDTH;
		m_runner->set_dimensions(m_ids[i], params);
		
		//TODO: change this to ENTITY_TANK
		m_runner->set_user_data(m_ids[i], 2);
		
		params.x = OFFSCREEN_X;
		params.y = OFFSCREEN_Y;
		m_runner->set_cur_pos(m_ids[i], params);
		
		next_bullet[i] = 0;
		
		for(int j = 0; j < BULLETS_PER_TANK; ++j){
			if(j >= m_tb->get_max_bullets()){
				continue; // don't allocate anymore
			}
			m_bullet[i][j] = m_tb->get_bullet();
		}
	}
}

void BasicTank::reset_phys_pointer(PhysRunner* p){
	m_runner = p;
}

void BasicTank::destroy(){
	for(int i = 0; i < MAX_TANKS; ++i){
		m_runner->destroy_object(m_ids[i]);
	}
}

void BasicTank::update(f32 dt){
	int idx = 0;
	#if __CUDA_ARCH__
		idx = threadIdx.x;
		if(idx < MAX_TANKS){
	#elif !defined(__CUDA_ARCH__)
		for(idx = 0; idx < MAX_TANKS; ++idx){
	#endif
			//TODO: code here for bounds checking
			// don't let the tank go beyond the walls
		}
}

void BasicTank::move_forward(tank_id tid){
	vec2 accel;
	f32 rot = util::degs_to_rads(m_runner->get_rotation(m_ids[tid]));
	accel.x = TANK_ACCEL_RATE * cosf(rot);
	accel.y = TANK_ACCEL_RATE * sinf(rot);
	m_runner->set_acceleration(m_ids[tid], accel);
	m_state[tid] = STATE_MOVING_FORWARD;
}

void BasicTank::move_backward(tank_id tid){
	vec2 accel;
	f32 rot = util::degs_to_rads(m_runner->get_rotation(m_ids[tid]));
	accel.x = -(TANK_ACCEL_RATE * cosf(rot));
	accel.y = -(TANK_ACCEL_RATE * sinf(rot));
	m_runner->set_acceleration(m_ids[tid], accel);
	m_state[tid] = STATE_MOVING_BACKWARD;
}

void BasicTank::stop(tank_id tid){
	m_state[tid] = STATE_NEUTRAL;
	vec2 accel;
	accel.x = accel.y = 0;
	m_runner->set_acceleration(m_ids[tid], accel);
	m_runner->set_cur_pos(m_ids[tid],
						  m_runner->get_cur_pos(m_ids[tid]));
}

void BasicTank::turn_left(tank_id tid){
	f32 rot = m_runner->get_rotation(m_ids[tid]);
	rot += TANK_ROT_RATE;
	if(rot >= 360){
		rot -= 360;
	}
	m_runner->set_rotation(m_ids[tid], rot);
	if(m_state[tid] == STATE_MOVING_FORWARD){
		move_forward(tid);
	}else if(m_state[tid] == STATE_MOVING_BACKWARD){
		move_backward(tid);
	}
}

void BasicTank::turn_right(tank_id tid){
	f32 rot = m_runner->get_rotation(m_ids[tid]);
	rot -= TANK_ROT_RATE;
	if(rot < 0){
		rot += 360;
	}
	m_runner->set_rotation(m_ids[tid], rot);
	if(m_state[tid] == STATE_MOVING_FORWARD){
		move_forward(tid);
	}else if(m_state[tid] == STATE_MOVING_BACKWARD){
		move_backward(tid);
	}
}

void BasicTank::fire(tank_id tid){
	bullet_id nb = next_bullet[tid]++;
	m_tb->fire_bullet(m_bullet[tid][nb],
					  m_runner->get_rotation(m_ids[tid]),
					  m_runner->get_cur_pos(m_ids[tid]));
	if(next_bullet[tid] >= BULLETS_PER_TANK){
		next_bullet[tid] = 0;
	}
}

tank_id BasicTank::spawn_tank(const vec2& pos, f32 rot){
	tank_id tid = m_next_tank++;
	m_runner->set_cur_pos(m_ids[tid], pos);
	m_runner->set_rotation(m_ids[tid], rot);
	return tid;
}

void BasicTank::kill_tank(tank_id tid){
	vec2 params;
	params.x = OFFSCREEN_X;
	params.y = OFFSCREEN_Y;
	m_runner->set_cur_pos(m_ids[tid], params);
	
	params.x = 0;
	params.y = 0;
	m_runner->set_acceleration(m_ids[tid], params);
}

vec2 BasicTank::get_tank_pos(tank_id tid){
	return m_runner->get_cur_pos(m_ids[tid]);
}

f32 BasicTank::get_tank_rot(tank_id tid){
	return m_runner->get_rotation(m_ids[tid]);
}