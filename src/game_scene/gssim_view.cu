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
#include <fstream>
#include <cuda.h>
#include "game_display.h"
#include "game_scene/gssim_view.h"
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

GSSimView::GSSimView(CL_GraphicContext& gc, CL_ResourceManager& resources,
					 sim_data& sd)
: m_physrunner(new Physics::PhysRunner::RunnerCore()),
m_simd(sd){
	
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
	m_testtank2.reset(new CL_Sprite(gc,
									"game_assets/tank_red",
									&resources));
	
	Physics::PhysRunner::initialize(m_physrunner.get());
	TankBullet::initialize(&m_bullets, m_physrunner.get());
	BasicTank::initialize(&m_tanks, m_physrunner.get(), &m_bullets);
	AI::initialize(&m_ai, &m_tanks, &m_bullets);

	// get the simulation data
	m_tanks = m_simd.tc;
	m_bullets = m_simd.bc;
	m_physrunner->bodies = m_simd.bodies[0];

	/*
	// stuff for cuda
	if(GameDisplay::s_usecuda){
		// allocate cuda memory
		cudaMalloc(reinterpret_cast<void**>(&m_cuda_runner),
				   sizeof(Physics::PhysRunner::RunnerCore));
		cudaMalloc(reinterpret_cast<void**>(&m_cuda_bullets),
				   sizeof(TankBullet::BulletCollection));
		cudaMalloc(reinterpret_cast<void**>(&m_cuda_tanks),
				   sizeof(BasicTank::TankCollection));
		cudaMalloc(reinterpret_cast<void**>(&m_cuda_ai),
				   sizeof(AI::AI_Core));
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
				   
		m_ai.bc = m_cuda_bullets;
		m_ai.tc = m_cuda_tanks;
		cudaMemcpy(m_cuda_ai, &m_ai,
				   sizeof(m_ai), cudaMemcpyHostToDevice);
	}
	*/
	m_frames_elapsed = 0;
}

GSSimView::~GSSimView(){
	BasicTank::destroy(&m_tanks);
	TankBullet::destroy(&m_bullets);
	/*
	cudaFree(m_cuda_tanks);
	cudaFree(m_cuda_bullets);
	cudaFree(m_cuda_runner);
	cudaFree(m_cuda_ai);
	*/
}

void GSSimView::onSceneDeactivate(){
}

void GSSimView::onSceneActivate(){
	m_timer.restart();
}

#include <iostream>

using namespace std;

void GSSimView::onFrameRender(CL_GraphicContext* gc){
	/*
	if(GameDisplay::s_usecuda){
		// re-assign the pointer to the CPU version
		BasicTank::reset_pointers(&m_tanks, m_physrunner.get(), &m_bullets);
		TankBullet::reset_phys_pointer(&m_bullets, m_physrunner.get());
	}
	*/
	
	// draw the background
	Physics::vec2 pos;
	apply_transform(gc, pos);
	m_background->draw(*gc, pos.x, pos.y);
	
	// draw the bullets, yes, we're cheating the numbers
	// OOP can wait another day
	for(int i = 0; i < 3; ++i){
		pos = TankBullet::get_bullet_pos(&m_bullets, i);
		apply_transform(gc, pos);
		if(m_bullets.state[i] != BULLET_STATE_INACTIVE){
			m_testbullet->draw(*gc, pos.x, pos.y);
		}
	}
	
	// draw the tanks
	if(m_tanks.state[m_playertank] != TANK_STATE_INACTIVE){
		pos = BasicTank::get_tank_pos(&m_tanks, m_playertank);
		apply_transform(gc, pos);
		f32 rot = BasicTank::get_tank_rot(&m_tanks, m_playertank);
		m_testtank->set_angle(CL_Angle(-rot, cl_degrees));
		m_testtank->draw(*gc, pos.x, pos.y);
	}

	if(m_tanks.state[m_player2tank] != TANK_STATE_INACTIVE){
		pos = BasicTank::get_tank_pos(&m_tanks, m_player2tank);
		apply_transform(gc, pos);
		f32 rot = BasicTank::get_tank_rot(&m_tanks, m_player2tank);
		m_testtank2->set_angle(CL_Angle(-rot, cl_degrees));
		m_testtank2->draw(*gc, pos.x, pos.y);
	}
	
	if(m_tanks.state[m_player3tank] != TANK_STATE_INACTIVE){
		pos = BasicTank::get_tank_pos(&m_tanks, m_player3tank);
		apply_transform(gc, pos);
		f32 rot = BasicTank::get_tank_rot(&m_tanks, m_player3tank);
		m_testtank2->set_angle(CL_Angle(-rot, cl_degrees));
		m_testtank2->draw(*gc, pos.x, pos.y);
	}
	
	// Debug info
	CL_StringFormat fmt("States: %1 %2 %3 %4 | Player pos: %5 %6 | Time elapsed: %7");
	fmt.set_arg(1, m_ai.bullet_vector[0]);
	fmt.set_arg(2, m_ai.tank_vector[0]);
	fmt.set_arg(3, m_ai.direction_state[0]);
	fmt.set_arg(4, m_ai.distance_state[0]);
	pos = Physics::PhysRunner::get_cur_pos(m_physrunner.get(),
														 m_tanks.phys_id[m_playertank]);
	fmt.set_arg(5, pos.x);
	fmt.set_arg(6, pos.y);
	fmt.set_arg(7, m_frames_elapsed);
	m_dbgmsg = fmt.get_result();
	m_debugfont->draw_text(*gc, 1, 12, m_dbgmsg, CL_Colorf::red);
}

/*
// #if __CUDA_ARCH__
__global__ void gsgame_step(f32 dt,
							tank_id player_tank,
							Physics::PhysRunner::RunnerCore* runner,
							TankBullet::BulletCollection* bullets,
							BasicTank::TankCollection* tanks,
							AI::AI_Core* aic){
	int idx = threadIdx.x;
	
	// AI operations
	AI::timestep(aic, dt);

	Physics::PhysRunner::timestep(runner, dt);
	TankBullet::update(bullets, dt);
	BasicTank::update(tanks, dt);
	
	// collision check
	if(idx < MAX_BULLETS){
		Collision::bullet_tank_check(bullets, tanks, idx);
	}
	if(idx < MAX_TANKS){
		Collision::tank_tank_check(tanks, idx);
	}
}
// #endif
*/

void GSSimView::onFrameUpdate(double dt,
						   CL_InputDevice* keyboard,
						   CL_InputDevice* mouse){
	// update the game state
	/*
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
												   m_cuda_ai);
						  
		// copy back the results for rendering
		cudaMemcpy(&m_bullets, m_cuda_bullets, sizeof(m_bullets),
				   cudaMemcpyDeviceToHost);
		cudaMemcpy(&m_tanks, m_cuda_tanks, sizeof(m_tanks),
				   cudaMemcpyDeviceToHost);
		cudaMemcpy(m_physrunner.get(), m_cuda_runner,
				   sizeof(Physics::PhysRunner::RunnerCore),
				   cudaMemcpyDeviceToHost);
		// not really needed but for debugging anyway
		cudaMemcpy(&m_ai, m_cuda_ai,
				   sizeof(AI::AI_Core),
				   cudaMemcpyDeviceToHost);
	}else{*/
		// process the player input
		if(m_tanks.state[m_playertank] != TANK_STATE_INACTIVE){

			// perform all the update
			//AI::timestep(&m_ai, dt);
			//Physics::PhysRunner::timestep(m_physrunner.get(), dt);
			// copy over the timestep data
			m_physrunner->bodies = m_simd.bodies[m_frames_elapsed];
			
			TankBullet::update(&m_bullets, dt);
			BasicTank::update(&m_tanks, dt);
			
			// perform collision detection for the bullets
			for(int i = 0; i < MAX_BULLETS; ++i){
				Collision::bullet_tank_check(&m_bullets, &m_tanks, i);
			}
			// perform collision detection for tanks
			for(int i = 0; i < MAX_TANKS; ++i){
				Collision::tank_tank_check(&m_tanks, i);
			}
		}
//	}
	
	// update the sprites
	m_background->update();
	m_testbullet->update();
	m_testtank->update();
	if(m_tanks.state[0] != TANK_STATE_INACTIVE){
		++m_frames_elapsed;
	}
}

