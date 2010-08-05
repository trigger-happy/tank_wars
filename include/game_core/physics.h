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

#define SHAPE_INVALID	0
#define SHAPE_CIRCLE	1
#define SHAPE_QUAD		2

struct physBody{
	physBody();
	
	vec2_array	old_pos;
	vec2_array	cur_pos;
	vec2_array	acceleration;
	f32 		rotation[MAX_ARRAY_SIZE];
	f32			max_vel[MAX_ARRAY_SIZE];
	
	bool		can_collide[MAX_ARRAY_SIZE];
	
	// for defining the shape of the object
	u32			shape_type[MAX_ARRAY_SIZE];
	
	// for custom code (to mark the object as a tank, bullet, etc)
	u32			user_data[MAX_ARRAY_SIZE];
	
	// x is width and y is height for quad
	// x is the radius and y is ignored for circle
	// totally irrelevant if SHAPE_INVALID
	vec2_array	dimension;
};

// forward declaration
class PhysRunner;

namespace PhysObject{
	
	u32 create_object(PhysRunner* pr);
	void destroy_object(PhysRunner* pr, u32 oid);
	
	vec2 get_cur_pos(PhysRunner* pr, u32 oid);
	vec2 get_acceleration(PhysRunner* pr, u32 oid);
	f32 get_rotation(PhysRunner* pr, u32 oid);
	f32 get_max_velocity(PhysRunner* pr, u32 oid);
	bool is_collidable(PhysRunner* pr, u32 oid);
	u32 get_shape_type(PhysRunner* pr, u32 oid);
	u32 get_user_data(PhysRunner* pr, u32 oid);
	vec2 get_dimensions(PhysRunner* pr, u32 oid);
	
	void set_cur_pos(PhysRunner* pr, u32 oid, const vec2& pos);
	void set_acceleration(PhysRunner* pr, u32 oid, const vec2& accel);
	void set_rotation(PhysRunner* pr, u32 oid, f32 r);
	void set_max_velocity(PhysRunner* pr, u32 oid, f32 mv);
	void set_shape_type(PhysRunner* pr, u32 oid, u32 st);
	void set_user_data(PhysRunner* pr, u32 oid, u32 ud);
	void set_dimensions(PhysRunner* pr, u32 oid, const vec2& dim);
	
	void should_collide(PhysRunner* pr, u32 oid, bool f);
	
};

class PhysRunner{
public:
	PhysRunner();
	~PhysRunner();
	void timestep(f32 dt);

	u32 get_slot();
	void free_slot(u32 id);
	void find_next_free_slot();
	
	// list of bodies. Not private due to CUDA limitations
	physBody					m_bodies;
	
private:
	//physBody*					m_pdevbodies;
	//physShape*				m_pdevshapes;
	//bool						m_update_dev_mem;
	std::bitset<MAX_ARRAY_SIZE>	m_free_slots;
	u32							m_first_free_slot;
};
	
}
#endif // PHYSICS_H
