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
#include "exports.h"

#define MAX_ARRAY_SIZE 256

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
	CUDA_EXPORT vec2_array();
	CUDA_EXPORT vec2 get_vec2(u32 id);
	
	f32 x[MAX_ARRAY_SIZE];
	f32 y[MAX_ARRAY_SIZE];
};

#define SHAPE_INVALID	0
#define SHAPE_CIRCLE	1
#define SHAPE_QUAD		2

typedef u32 pBody;
typedef u32 pShape;


struct physBody{
	CUDA_EXPORT physBody();
	
	vec2_array	old_pos;
	vec2_array	cur_pos;
	vec2_array	acceleration;
	f32 		rotation[MAX_ARRAY_SIZE];
	f32			max_vel[MAX_ARRAY_SIZE];
	
	bool		can_collide[MAX_ARRAY_SIZE];
	
	// for defining the shape of the object
	pShape	 	shape_type[MAX_ARRAY_SIZE];
	
	// for custom code (to mark the object as a tank, bullet, etc)
	u32			user_data[MAX_ARRAY_SIZE];
	
	// x is width and y is height for quad
	// x is the radius and y is ignored for circle
	// totally irrelevant if SHAPE_INVALID
	vec2_array	dimension;
};


class PhysRunner{
public:
	CUDA_HOST PhysRunner();
	CUDA_HOST ~PhysRunner();
	CUDA_EXPORT void timestep(f32 dt);

	
	CUDA_EXPORT pBody create_object();
	CUDA_EXPORT void destroy_object(pBody bd);
	
	CUDA_EXPORT vec2 get_cur_pos(pBody bd);
	CUDA_EXPORT vec2 get_acceleration(pBody bd);
	CUDA_EXPORT f32 get_rotation(pBody bd);
	CUDA_EXPORT f32 get_max_velocity(pBody bd);
	CUDA_EXPORT bool is_collidable(pBody bd);
	CUDA_EXPORT pShape get_shape_type(pBody bd);
	CUDA_EXPORT u32 get_user_data(pBody bd);
	CUDA_EXPORT vec2 get_dimensions(pBody bd);
	
	CUDA_EXPORT void set_cur_pos(pBody bd, const vec2& pos);
	CUDA_EXPORT void set_acceleration(pBody bd, const vec2& accel);
	CUDA_EXPORT void set_rotation(pBody bd, f32 r);
	CUDA_EXPORT void set_max_velocity(pBody bd, f32 mv);
	CUDA_EXPORT void set_shape_type(pBody bd, pShape st);
	CUDA_EXPORT void set_user_data(pBody bd, u32 ud);
	CUDA_EXPORT void set_dimensions(pBody bd, const vec2& dim);
	
	CUDA_EXPORT void should_collide(pBody bd, bool f);
	
private:
	CUDA_EXPORT u32 get_slot();
	CUDA_EXPORT void free_slot(u32 id);
	CUDA_EXPORT void find_next_free_slot();
	
private:
	physBody					m_bodies;
	//physBody*					m_pdevbodies;
	//physShape*				m_pdevshapes;
	//bool						m_update_dev_mem;
	//TODO: change this to a custom bitvector
	u8							m_free_slots[MAX_ARRAY_SIZE];
	u32							m_first_free_slot;
};
	
}
#endif // PHYSICS_H
