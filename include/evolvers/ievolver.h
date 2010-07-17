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

#ifndef IEVOLVER_H
#define IEVOLVER_H

template<typename Derived>
class iEvolver{
public:
	/*!
	Initialize the evolver
	*/
	void initialize(){
		static_cast<Derived*>(this)->initialize_impl();
	}
	
	/*!
	Perform cleanup operations
	*/
	void cleanup(){
		static_cast<Derived*>(this)->cleanup_impl();
	}
	
	/*!
	Iterate through a single timestep in the simulation.
	\param dt - The delta time in seconds between timesteps
	\note Each call will simulate a single frame in ALL game instances
	*/
	void frame_step(float dt){
		static_cast<Derived*>(this)->frame_step_impl(dt);
	}
	
	/*!
	Retrieve the current game state
	*/
	void retrieve_state(){
		static_cast<Derived*>(this)->retrieve_state_impl();
	}
};

#endif //IEVOLVER_H