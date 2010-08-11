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
#include "../game_core/twgame.cu"
// #include "game_core/physics.h"
// #include "game_core/tankbullet.h"
// #include "game_core/basictank.h"

#define SCALE_FACTOR 12

#define CUDA_BLOCKS 1
#define CUDA_THREADS MAX_ARRAY_SIZE

// helper function
void apply_transform(CL_GraphicContext* gc, Physics::vec2& c){
	c.x *= SCALE_FACTOR;
	c.y *= -SCALE_FACTOR;
	c.x += gc->get_width()/2;
	c.y += gc->get_height()/2;
}

GSGame::GSGame(CL_GraphicContext& gc, CL_ResourceManager& resources)
: m_physrunner(new Physics::PhysRunner::RunnerCore()){
	
	// setup the debug text
	CL_FontDescription desc;
	desc.set_typeface_name("monospace");
	desc.set_height(12);
	m_debugfont.reset(new CL_Font_System(gc, desc));
	
	m_background.reset(new CL_Sprite(gc,
									 "game_assets/background",
									 &resources));
	m_testbullet.reset(new CL_Sprite(gc,
									 "game_assets/bullet",
									 &resources));
	m_testtank.reset(new CL_Sprite(gc,
								   "game_assets/tank_blu",
								   &resources));
	
	TankBullet::initialize(&m_bullets, m_physrunner.get());
	BasicTank::initialize(&m_tanks, m_physrunner.get(), &m_bullets);
	
	// test code
	Physics::vec2 params;
	m_playertank = BasicTank::spawn_tank(&m_tanks, params, 0);
	m_player_input = 0;
	
	// stuff for cuda
	if(GameDisplay::s_usecuda){
		// allocate cuda memory
		cudaMalloc(reinterpret_cast<void**>(&m_cuda_runner),
				   sizeof(Physics::PhysRunner::RunnerCore));
		cudaMalloc(reinterpret_cast<void**>(&m_cuda_bullets),
				   sizeof(TankBullet::BulletCollection));
		cudaMalloc(reinterpret_cast<void**>(&m_cuda_tanks),
				   sizeof(BasicTank::TankCollection));
// 		cudaMalloc(reinterpret_cast<void**>(&m_cuda_player_input),
// 				   sizeof(m_player_input));
		
		// reset the pointers
		BasicTank::reset_pointers(&m_tanks, m_cuda_runner, m_cuda_bullets);
		TankBullet::reset_phys_pointer(&m_bullets, m_cuda_runner);
		
		// copy over the stuff to GPU mem
		cudaMemcpy(m_cuda_runner, m_physrunner.get(),
				   sizeof(Physics::PhysRunner::RunnerCore), cudaMemcpyHostToDevice);
				   
		cudaMemcpy(m_cuda_bullets, &m_bullets,
				   sizeof(m_bullets), cudaMemcpyHostToDevice);
				   
		cudaMemcpy(m_cuda_tanks, &m_tanks,
				   sizeof(m_tanks), cudaMemcpyHostToDevice);
// 		cudaMemcpy(m_cuda_player_input, &m_player_input,
// 				   sizeof(m_player_input), cudaMemcpyHostToDevice);
	}
}

GSGame::~GSGame(){
	BasicTank::destroy(&m_tanks);
	TankBullet::destroy(&m_bullets);
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
		BasicTank::reset_pointers(&m_tanks, m_physrunner.get(), &m_bullets);
		TankBullet::reset_phys_pointer(&m_bullets, m_physrunner.get());
	}
	// draw the background
	Physics::vec2 pos;
	apply_transform(gc, pos);
	m_background->draw(*gc, pos.x, pos.y);
	
	// draw the bullets
	pos = TankBullet::get_bullet_pos(&m_bullets, 0);
	apply_transform(gc, pos);
	m_testbullet->draw(*gc, pos.x, pos.y);
	
	// draw the tanks
	pos = BasicTank::get_tank_pos(&m_tanks, m_playertank);
	apply_transform(gc, pos);
	f32 rot = BasicTank::get_tank_rot(&m_tanks, m_playertank);
	m_testtank->set_angle(CL_Angle(-rot, cl_degrees));
	m_testtank->draw(*gc, pos.x, pos.y);
	
	// Debug info
// 	CL_StringFormat fmt("S: %1 X: %2 Y: %3 Rot: %4 AX: %5 AY: %6 XV: %7 XY: %8");
// 	switch(m_tanks.state[m_playertank]){
// 		case STATE_INACTIVE:
// 			fmt.set_arg(1, "Inactive");
// 			break;
// 		case STATE_NEUTRAL:
// 			fmt.set_arg(1, "Neutral");
// 			break;
// 		case STATE_MOVING_FORWARD:
// 			fmt.set_arg(1, "Forward");
// 			break;
// 		case STATE_MOVING_BACKWARD:
// 			fmt.set_arg(1, "Backward");
// 			break;
// 		case STATE_FIRING:
// 			fmt.set_arg(1, "Firing");
// 			break;
// 		case STATE_RELOADING:
// 			fmt.set_arg(1, "Reloading");
// 			break;
// 	}
// 	fmt.set_arg(2, pos.x);
// 	fmt.set_arg(3, pos.y);
// 	fmt.set_arg(4, BasicTank::get_tank_rot(&m_tanks, m_playertank));
// 	Physics::vec2 accel = BasicTank::get_tank_accel(&m_tanks, m_playertank);
// 	fmt.set_arg(5, accel.x);
// 	fmt.set_arg(6, accel.y);
// 	Physics::pBody pb = m_tanks.phys_id[m_playertank];
// 	accel.y = m_physrunner->bodies.cur_pos.y[pb] - m_physrunner->bodies.old_pos.y[pb];
// 	accel.x = m_physrunner->bodies.cur_pos.x[pb] - m_physrunner->bodies.old_pos.x[pb];
// 	fmt.set_arg(7, accel.x);
// 	fmt.set_arg(8, accel.y);
// 	m_dbgmsg = fmt.get_result();
// 	m_debugfont->draw_text(*gc, 1, 12, m_dbgmsg, CL_Colorf::red);
}


// #if __CUDA_ARCH__
__global__ void gsgame_step(f32 dt,
							tank_id player_tank,
							Physics::PhysRunner::RunnerCore* runner,
							TankBullet::BulletCollection* bullets,
							BasicTank::TankCollection* tanks,
							u32 player_input){
	int idx = threadIdx.x;
	
	//TODO: perform AI operations here
	
	if(idx == 0){
		// thread 0 will perform the input processing
		// all other threads will do squat (wasteful but can't be helped)
		if(player_input & PLAYER_FIRE){
			BasicTank::fire(tanks, player_tank);
		}
		
		if(player_input & PLAYER_STOP){
			BasicTank::stop(tanks, player_tank);
		}else if(player_input & PLAYER_FORWARD){
			BasicTank::move_forward(tanks, player_tank);
		}else if(player_input & PLAYER_BACKWARD){
			BasicTank::move_backward(tanks, player_tank);
		}
		
		if(player_input & PLAYER_LEFT){
			BasicTank::turn_left(tanks, player_tank);
		}else if(player_input & PLAYER_RIGHT){
			BasicTank::turn_right(tanks, player_tank);
		}
	}
	Physics::PhysRunner::timestep(runner, dt);
	TankBullet::update(bullets, dt);
	BasicTank::update(tanks, dt);
}
// #endif

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
	}else if(keyboard->get_keycode(CL_KEY_UP)){
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
		cudaMemcpy(m_physrunner.get(), m_cuda_runner,
				   sizeof(Physics::PhysRunner::RunnerCore),
				   cudaMemcpyDeviceToHost);
	}else{
		// process the player input
		if(m_player_input & PLAYER_FIRE){
			BasicTank::fire(&m_tanks, m_playertank);
		}
		
		if(m_player_input & PLAYER_STOP){
			BasicTank::stop(&m_tanks, m_playertank);
		}
		
		if(m_player_input & PLAYER_FORWARD){
			BasicTank::move_forward(&m_tanks, m_playertank);
		}else if(m_player_input & PLAYER_BACKWARD){
			BasicTank::move_backward(&m_tanks, m_playertank);
		}
		
		if(m_player_input & PLAYER_LEFT){
			BasicTank::turn_left(&m_tanks, m_playertank);
		}else if(m_player_input & PLAYER_RIGHT){
			BasicTank::turn_right(&m_tanks, m_playertank);
		}
		
		// perform all the update
		Physics::PhysRunner::timestep(m_physrunner.get(), dt);
		TankBullet::update(&m_bullets, dt);
		BasicTank::update(&m_tanks, dt);
	}
	
	// update the sprites
	m_background->update();
	m_testbullet->update();
	m_testtank->update();
}

