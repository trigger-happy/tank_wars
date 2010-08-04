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

#ifndef PHYSICS_H
#define PHYSICS_H
#include <cuda.h>
#include <bitset>
#include "types.h"

#define MAX_ARRAY_SIZE 256
#define MAX_VERTICES 4

namespace Physics{
	
struct vec2{
	vec2(){
		x = 0;
		y = 0;
	}
	
	f32 x;
	f32 y;
};

struct vec2_array{
	vec2_array();
	vec2 get_vec2(u32 id);
	
	f32 x[MAX_ARRAY_SIZE];
	f32 y[MAX_ARRAY_SIZE];
};

struct physBody{
	physBody();
	
	vec2_array	old_pos;
	vec2_array	cur_pos;
	vec2_array	acceleration;
	f32 		rotation[MAX_ARRAY_SIZE];
	f32			max_vel[MAX_ARRAY_SIZE];
	
	bool		can_collide[MAX_ARRAY_SIZE];
};


struct physShape{
	// if radius is 0, it's a quad, if it's greater, it's a circle
	f32 radius[MAX_ARRAY_SIZE];
	f32 vertices[MAX_ARRAY_SIZE][MAX_VERTICES];
};

// forward declaration
class PhysRunner;

class PhysObject{
public:
	PhysObject(PhysRunner* p);
	~PhysObject();
	
	virtual void on_physics_update() = 0;
	
	vec2 get_cur_pos();
	vec2 get_acceleration();
	f32 get_rotation();
	f32 get_max_velocity();
	bool is_collidable();
	
	void set_cur_pos(const vec2& pos);
	void set_acceleration(const vec2& accel);
	void set_rotation(f32 r);
	void set_max_velocity(f32 mv);
	
	void should_collide(bool f);
	
private:
	PhysRunner*	m_runner;
	u32			m_objid;
};

class PhysRunner{
public:
	PhysRunner();
	~PhysRunner();
	void initialize();
	void timestep(f32 dt);
	
	inline void update_dev_mem(){
		m_update_dev_mem = true;
	}
	
private:
	void copy_from_device();
	void copy_to_device();
	
private:
	friend class PhysObject;
	u32 get_slot();
	void free_slot(u32 id);
	void find_next_free_slot();
	
private:
	physBody					m_hostbodies;
	physShape					m_hostshapes;
	physBody*					m_pdevbodies;
	physShape*					m_pdevshapes;
	bool						m_update_dev_mem;
	std::bitset<MAX_ARRAY_SIZE>	m_free_slots;
	u32							m_first_free_slot;
};
	
}
#endif // PHYSICS_H
