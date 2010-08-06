/*
    <one line to give the library's name and an idea of what it does.>
    Copyright (C) <year>  <name of author>

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/
#include "game_display.h"
#include "game_scene/gsgame.h"
#include "game_core/physics.h"
#include "game_core/tankbullet.h"
#include "game_core/basictank.h"

#define SCALE_FACTOR 12

#define CUDA_BLOCKS 1
#define CUDA_THREADS MAX_ARRAY_SIZE

//NOTE: external calls are not supported, cheap hack but it works
#if __CUDA_ARCH__
#include "../game_core/physics.cu"
#include "../game_core/tankbullet.cu"
#include "../game_core/basictank.cu"
#endif

using namespace Physics;

// helper function
void apply_transform(CL_GraphicContext* gc, vec2& c){
	c.x *= SCALE_FACTOR;
	c.y *= -SCALE_FACTOR;
	c.x += gc->get_width()/2;
	c.y += gc->get_height()/2;
}

GSGame::GSGame(CL_GraphicContext& gc, CL_ResourceManager& resources)
: m_physrunner(new PhysRunner()){
	
	m_background.reset(new CL_Sprite(gc,
									 "game_assets/background",
									 &resources));
	m_testbullet.reset(new CL_Sprite(gc,
									 "game_assets/bullet",
									 &resources));
	m_testtank.reset(new CL_Sprite(gc,
								   "game_assets/tank_blu",
								   &resources));
								   
	m_bullets.initialize(m_physrunner.get());
	m_tanks.initialize(m_physrunner.get(), &m_bullets);
	
	// test code
	vec2 params;
	m_playertank = m_tanks.spawn_tank(params, 0);
	m_player_input = 0;
	
	// stuff for cuda
	if(GameDisplay::s_usecuda){
		// allocate cuda memory
		cudaMalloc(reinterpret_cast<void**>(&m_cuda_runner),
				   sizeof(PhysRunner));
		cudaMalloc(reinterpret_cast<void**>(&m_cuda_bullets),
				   sizeof(TankBullet));
		cudaMalloc(reinterpret_cast<void**>(&m_cuda_tanks),
				   sizeof(BasicTank));
// 		cudaMalloc(reinterpret_cast<void**>(&m_cuda_player_input),
// 				   sizeof(m_player_input));
		
		// reset the pointers
		m_bullets.reset_phys_pointer(m_cuda_runner);
		m_tanks.reset_phys_pointer(m_cuda_runner);
		
		// copy over the stuff to GPU mem
		cudaMemcpy(m_cuda_runner, m_physrunner.get(),
				   sizeof(PhysRunner), cudaMemcpyHostToDevice);
				   
		cudaMemcpy(m_cuda_bullets, &m_bullets,
				   sizeof(m_bullets), cudaMemcpyHostToDevice);
				   
		cudaMemcpy(m_cuda_tanks, &m_tanks,
				   sizeof(m_tanks), cudaMemcpyHostToDevice);
// 		cudaMemcpy(m_cuda_player_input, &m_player_input,
// 				   sizeof(m_player_input), cudaMemcpyHostToDevice);
	}
}

GSGame::~GSGame(){
	m_tanks.destroy();
	m_bullets.destroy();
	cudaFree(m_cuda_tanks);
	cudaFree(m_cuda_bullets);
	cudaFree(m_cuda_runner);
}

void GSGame::onSceneDeactivate(){
}

void GSGame::onSceneActivate(){
}

void GSGame::onFrameRender(CL_GraphicContext* gc){
	if(GameDisplay::s_usecuda){
		// re-assign the pointer to the CPU version
		m_tanks.reset_phys_pointer(m_physrunner.get());
		m_bullets.reset_phys_pointer(m_physrunner.get());
	}
	// draw the background
	vec2 pos;
	apply_transform(gc, pos);
	m_background->draw(*gc, pos.x, pos.y);
	
	// draw the bullets
	pos = m_bullets.get_bullet_pos(0);
	apply_transform(gc, pos);
	m_testbullet->draw(*gc, pos.x, pos.y);
	
	// draw the tanks
	pos = m_tanks.get_tank_pos(m_playertank);
	apply_transform(gc, pos);
	f32 rot = m_tanks.get_tank_rot(m_playertank);
	m_testtank->set_angle(CL_Angle(-rot, cl_degrees));
	m_testtank->draw(*gc, pos.x, pos.y);
}

__global__ void gsgame_step(f32 dt,
							tank_id player_tank,
							PhysRunner* runner,
							TankBullet* bullets,
							BasicTank* tanks,
							u8 player_input){
	int idx = threadIdx.x;
	
	//TODO: perform AI operations here
	
	if(idx == 0){
		// thread 0 will perform the input processing
		// all other threads will do squat (wasteful but can't be helped)
		if(player_input & PLAYER_FIRE){
			tanks->fire(player_tank);
		}
		
		if(player_input & PLAYER_STOP){
			tanks->stop(player_tank);
		}
		
		if(player_input & PLAYER_FORWARD){
			tanks->move_forward(player_tank);
		}else if(player_input & PLAYER_BACKWARD){
			tanks->move_backward(player_tank);
		}
		
		if(player_input & PLAYER_LEFT){
			tanks->turn_left(player_tank);
		}else if(player_input & PLAYER_RIGHT){
			tanks->turn_right(player_tank);
		}
	}
 	runner->timestep(dt);
 	bullets->update(dt);
 	tanks->update(dt);
}

void GSGame::onFrameUpdate(double dt,
						   CL_InputDevice* keyboard,
						   CL_InputDevice* mouse){
	//TODO: code here for AI update
	
	// keyboard control processing
	// test code for the time being
	m_player_input = 0;
	if(keyboard->get_keycode(CL_KEY_SPACE)){
		//m_tanks.fire(m_playertank);
		m_player_input |= PLAYER_FIRE;
	}
	
	if(keyboard->get_keycode(CL_KEY_RETURN)){
		//m_tanks.stop(m_playertank);
		m_player_input |= PLAYER_STOP;
	}
	
	if(keyboard->get_keycode(CL_KEY_UP)){
		//m_tanks.move_forward(m_playertank);
		m_player_input |= PLAYER_FORWARD;
	}else if(keyboard->get_keycode(CL_KEY_DOWN)){
		//m_tanks.move_backward(m_playertank);
		m_player_input |= PLAYER_BACKWARD;
	}
	
	if(keyboard->get_keycode(CL_KEY_LEFT)){
		//m_tanks.turn_left(m_playertank);
		m_player_input |= PLAYER_LEFT;
	}else if(keyboard->get_keycode(CL_KEY_RIGHT)){
		//m_tanks.turn_right(m_playertank);
		m_player_input |= PLAYER_RIGHT;
	}
	
	// update the game state
	if(GameDisplay::s_usecuda){
		// copy over the player input
// 		cudaMemcpy(m_cuda_player_input, &m_player_input,
// 				   sizeof(m_player_input), cudaMemcpyHostToDevice);

		// call the update
		gsgame_step<<<CUDA_BLOCKS, CUDA_THREADS>>>(dt,
												   m_playertank,
												   m_cuda_runner,
												   m_cuda_bullets,
												   m_cuda_tanks,
												   m_player_input);
						  
		// copy back the results for rendering
		cudaMemcpy(&m_bullets, m_cuda_bullets, sizeof(m_bullets),
				   cudaMemcpyDeviceToHost);
		cudaMemcpy(&m_tanks, m_cuda_tanks, sizeof(m_tanks),
				   cudaMemcpyDeviceToHost);
		cudaMemcpy(m_physrunner.get(), m_cuda_runner, sizeof(m_physrunner),
				   cudaMemcpyDeviceToHost);
	}else{
		// process the player input
		if(m_player_input & PLAYER_FIRE){
			m_tanks.fire(m_playertank);
		}
		
		if(m_player_input & PLAYER_STOP){
			m_tanks.stop(m_playertank);
		}
		
		if(m_player_input & PLAYER_FORWARD){
			m_tanks.move_forward(m_playertank);
		}else if(m_player_input & PLAYER_BACKWARD){
			m_tanks.move_backward(m_playertank);
		}
		
		if(m_player_input & PLAYER_LEFT){
			m_tanks.turn_left(m_playertank);
		}else if(m_player_input & PLAYER_RIGHT){
			m_tanks.turn_right(m_playertank);
		}
		
		// perform all the update
		m_physrunner->timestep(dt);
		m_bullets.update(dt);
		m_tanks.update(dt);
	}
	
	// update the sprites
	m_background->update();
	m_testbullet->update();
	m_testtank->update();
}

