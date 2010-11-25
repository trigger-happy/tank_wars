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
#include <bitset>
#include "types.h"
#include "exports.h"

#define MAX_ARRAY_SIZE	32
#define OFFSCREEN_X		-33
#define OFFSCREEN_Y		25
#define INVALID_ID		MAX_ARRAY_SIZE+1

namespace Physics{
	
/*!
Simple vector class for floats
*/
template<typename T>
struct vec2{
	CUDA_EXPORT vec2(){
		x = 0;
		y = 0;
	}
	
	CUDA_EXPORT void normalize(){
		T l = length();
		x /= l;
		y /= l;
	}
	
	CUDA_EXPORT T length(){
		return sqrt(static_cast<f32>(x*x) + static_cast<f32>(y*y));
	}
	
	CUDA_EXPORT vec2 operator-=(const vec2& rhs){
		vec2 temp;
		x -= rhs.x;
		y -= rhs.y;
		temp.x = x;
		temp.y = y;
		return temp;
	}
	
	CUDA_EXPORT T operator*(const vec2& rhs){
		T temp = (x * rhs.x) + (y * rhs.y);
		return temp;
	}
	
	CUDA_EXPORT vec2 operator-() const{
		vec2 temp = *this;
		temp.x *= -1;
		temp.y *= -1;
		return temp;
	}
	
	T x;
	T y;
};

/*!
Structure of arrays for a vec2 class
Purpose of this is to improve memory coallescing on the gpu
*/
template<typename T>
struct vec2_array{
	CUDA_EXPORT vec2_array(){
		#if __CUDA_ARCH__
		// device code
		for(int i = 0; i < MAX_ARRAY_SIZE; ++i){
			x[i] = 0;
			y[i] = 0;
		}
		#elif !defined(__CUDA_ARCH__)
		// host code
		std::fill(x, x + MAX_ARRAY_SIZE, 0);
		std::fill(y, y + MAX_ARRAY_SIZE, 0);
		#endif
	}
	
	CUDA_EXPORT vec2<T> get_vec2(u32 id){
		vec2<T> temp;
		temp.x = x[id];
		temp.y = y[id];
		return temp;
	}
	
	CUDA_EXPORT void normalize(u32 id){
		f32 l = sqrt(x[id]*x[id] + y[id]*y[id]);
		x[id] /= l;
		y[id] /= l;
	}
	
	T x[MAX_ARRAY_SIZE];
	T y[MAX_ARRAY_SIZE];
};

#define SHAPE_INVALID	0
#define SHAPE_CIRCLE	1
#define SHAPE_QUAD		2

typedef u32 pBody;
typedef u32 pShape;

/*!
Structure of array for the physics body objects. An object's parameters
can be accessed grabbing the nth element in any of the array of parameters.
pBody is used as the index for any given object.
*/
struct physBody{
	vec2_array<s32>	old_pos;
	vec2_array<s32>	cur_pos;
	vec2_array<f32>	acceleration;
	f32 			rotation[MAX_ARRAY_SIZE];
	f32				max_vel[MAX_ARRAY_SIZE];
	
	bool			can_collide[MAX_ARRAY_SIZE];
	
	// for defining the shape of the object
	pShape	 		shape_type[MAX_ARRAY_SIZE];
	
	// for custom code (to mark the object as a tank, bullet, etc)
	u32				user_data[MAX_ARRAY_SIZE];
	
	// x is width and y is height for quad
	// x is the radius and y is ignored for circle
	// totally irrelevant if SHAPE_INVALID
	vec2_array<s32>		dimension;
};

	CUDA_EXPORT void init_physbody(Physics::physBody* pb);


namespace PhysRunner{
	
	/*!
	Context object for the physics runner.
	Contains the physBody structure of arrays and an array to denote free
	slots for the physics objects.
	*/
	struct RunnerCore{
		physBody					bodies;
		//TODO: change this to a custom bitvector
		u8							free_slots[MAX_ARRAY_SIZE];
		u32							first_free_slot;
	};
	
	/*!
	Initialize the RunnerCore object
	\param rc The RunnerCore object 
	*/
	CUDA_HOST void initialize(RunnerCore* rc);
	
	/*!
	Cleanup the RunnerCore object
	\param rc The RunnerCore object
	*/
	CUDA_HOST void cleanup(RunnerCore* rc);
	
	/*!
	Perform a frame update
	\param rc The RunnerCore object.
	\param dt The frame time in seconds.
	*/
	CUDA_EXPORT void timestep(RunnerCore* rc, f32 dt);

	
	/*!
	Create a physics object by reserving a slot in the PhysBody lists
	\param rc The RunnerCore object.
	\return The index in the physics body list.
	*/
	CUDA_EXPORT pBody create_object(RunnerCore* rc);
	
	/*!
	Destroy the physics object so that the slot may be used elsewhere
	\param rc The RunnerCore object.
	\param oid The index of the object.
	*/
	CUDA_EXPORT void destroy_object(RunnerCore* rc, pBody oid);
	
	/*!
	Get the current position of the physics object.
	\param rc The RunnerCore object
	\param oid The index of the physicsbody
	\return A vec2 object containing the current position
	*/
	CUDA_EXPORT vec2<s32> get_cur_pos(RunnerCore* rc, pBody oid);
	
	/*!
	Get the previous position of the physics object.
	\param rc The RunnerCore object
	\param oid The index of the physicsbody
	\return A vec2 object containing the previous position
	*/
	CUDA_EXPORT vec2<s32> get_prev_pos(RunnerCore* rc, pBody oid);
	
	/*!
	Get the acceleration of the object
	\param rc The RunnerCore object
	\param oid The index of the physicsbody
	\return A vec2 containing the acceleration of the object.
	*/
	CUDA_EXPORT vec2<f32> get_acceleration(Physics::PhysRunner::RunnerCore* rc, pBody oid);
	
	/*!
	Get the rotation of the physics object
	\param rc The RunnerCore object.
	\param oid The index of the physicsbody
	\return The rotation of the object.
	*/
	CUDA_EXPORT f32 get_rotation(Physics::PhysRunner::RunnerCore* rc, pBody oid);
	
	/*!
	Get the maximum velocity of the object.
	\param rc The RunnerCore object.
	\param oid The index of the physicsbody.
	\return The maximum velocity of the object.
	*/
	CUDA_EXPORT f32 get_max_velocity(RunnerCore* rc, pBody oid);
	
	/*!
	Returns true if the object may collides, false otherwise
	\param rc The RunnerCore object.
	\param oid The index of the physicsbody.
	\return A boolean
	*/
	CUDA_EXPORT bool is_collidable(RunnerCore* rc, pBody oid);
	
	/*!
	Get the shape type of the physics object.
	\param rc The RunnerCore object.
	\param oid The index of the physicsbody.
	\return The shape type of the object.
	*/
	CUDA_EXPORT pShape get_shape_type(RunnerCore* rc, pBody oid);
	
	/*!
	Get any attached user data on the physics object.
	\param rc The RunnerCore object.
	\param oid The index of the physicsbody.
	\return The attached user data.
	*/
	CUDA_EXPORT u32 get_user_data(RunnerCore* rc, pBody oid);
	
	/*!
	Get the dimensions of the physics object.
	\param rc The RunnerCore object.
	\param oid The index of the physicsbody.
	\return A vec2 object containing the physical dimensions of the object.
	*/
	CUDA_EXPORT vec2<s32> get_dimensions(RunnerCore* rc, pBody oid);
	
	
	/*!
	Set the current position of the physics object.
	\param rc The RunnerCore object.
	\param oid The index of the physicsbody.
	\param pos The new object position.
	*/
	CUDA_EXPORT void set_cur_pos(RunnerCore* rc, pBody oid, const vec2<s32>& pos);
	
	/*!
	Set the acceleration of the physics object.
	\param rc The RunnerCore object.
	\param oid The index of the physicsbody.
	\param accel The acceleration of the physicsobject.
	*/
	CUDA_EXPORT void set_acceleration(RunnerCore* rc,
									  pBody oid, const vec2<f32>& accel);
									  
	/*!
	Set the rotation of the object.
	\param rc The RunnerCore object.
	\param oid The index of the physicsbody.
	\param r The rotation of the object.
	*/
	CUDA_EXPORT void set_rotation(RunnerCore* rc, pBody oid, f32 r);
	
	/*!
	Set the maximum velocity of the object.
	\param rc The RunnerCore object.
	\param oid The index of the physicsbody.
	\param mv The maximum velocity.
	*/
	CUDA_EXPORT void set_max_velocity(RunnerCore* rc, pBody oid, f32 mv);
	
	/*!
	Set the shape type of the object.
	\param rc The RunnerCore object.
	\param oid The index of the physicsbody.
	\param st The shape type.
	*/
	CUDA_EXPORT void set_shape_type(RunnerCore* rc, pBody oid, pShape st);
	
	/*!
	Attach some user data on the object.
	\param rc The RunnerCore object.
	\param oid The index of the physicsbody.
	\param ud The user data.
	*/
	CUDA_EXPORT void set_user_data(RunnerCore* rc, pBody oid, u32 ud);
	
	/*!
	Set the dimensions of the object.
	\param rc The RunnerCore object.
	\param oid The index of the physicsbody.
	\param dim The dimensions of the object.
	*/
	CUDA_EXPORT void set_dimensions(RunnerCore* rc, pBody oid, const vec2<s32>& dim);
	
	/*!
	Set the collidable property of an object.
	\param rc The RunnerCore object.
	\param oid The index of the physicsbody.
	\param f true if it collidable, false otherwise.
	*/
	CUDA_EXPORT void should_collide(RunnerCore* rc, pBody oid, bool f);
	
	/*!
	Get a slot from the PhysBody array.
	\note INTERNAL USE ONLY
	\param rc The RunnerCore object.
	*/
	CUDA_EXPORT u32 get_slot(RunnerCore* rc);
	
	/*!
	Free a slot from the PhysBody array.
	\note INTERNAL USE ONLY
	\param rc The RunnerCore object.
	\param id The id to free
	*/
	CUDA_EXPORT void free_slot(RunnerCore* rc, u32 id);
	
	/*!
	Get the next available slot for physics objects.
	\note INTERNAL USE ONLY.
	\param rc The RunnerCore object.
	*/
	CUDA_EXPORT void find_next_free_slot(RunnerCore* rc);
	
	/*!
	Get the velocity of the object
	\param rc The RunnerCore object
	\param id The id of the phys object
	\return The velocity of the object
	*/
	CUDA_EXPORT f32 get_cur_velocity(RunnerCore* rc, pBody oid);
	
	/*!
	Get the velocity vector of the object
	\param rc The RunnerCore object
	\param id The id of the phys object
	\return The velocity vector of the object
	*/
	vec2<s32> get_velocity_vector(Physics::PhysRunner::RunnerCore* rc, Physics::pBody oid);
}
	
}
#endif // PHYSICS_H
