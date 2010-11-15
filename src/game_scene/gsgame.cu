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
#include <ClanLib/Core/XML/dom_element.h>
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

void GSGame::load_shaders(CL_GraphicContext& gc, CL_ResourceManager& res){
	CL_String vfile = "resources/";
	CL_String ffile = "resources/";
	
	CL_Resource shader_resource = res.get_resource("shaders/vertex");
	vfile += shader_resource.get_element().get_attribute("file");
	shader_resource = res.get_resource("shaders/fragment");
	ffile += shader_resource.get_element().get_attribute("file");
	
	m_shader = CL_ProgramObject::load(gc, vfile, ffile);
	m_shader.bind_attribute_location(0, "Position");
	m_shader.bind_attribute_location(1, "Color0");
	m_shader.bind_attribute_location(2, "TexCoord0");
	if(!m_shader.link()){
		throw CL_Exception("Shader Link Fail"+m_shader.get_info_log());
	}
}

GSGame::GSGame(CL_GraphicContext& gc, CL_ResourceManager& resources)
: m_physrunner(new Physics::PhysRunner::RunnerCore()), m_offscreen_buffer(gc),
m_offscreen_texture(gc, gc.get_width(), gc.get_height()){
	
	m_offscreen_texture.set_min_filter(cl_filter_nearest);
	m_offscreen_texture.set_mag_filter(cl_filter_nearest);
	m_offscreen_buffer.attach_color_buffer(0, m_offscreen_texture);
	
	load_shaders(gc, resources);
	
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
	
	// test code
	Physics::vec2 params;
	params.x = -25;
	m_playertank = BasicTank::spawn_tank(&m_tanks, params, 90, 0);
	params.x = 3;
	params.y = 12;
	m_player2tank = BasicTank::spawn_tank(&m_tanks, params, 180, 1);
	params.x = 20;
	params.y = -12;
	m_player3tank = BasicTank::spawn_tank(&m_tanks, params, 180, 1);
	m_player_input = 0;
	m_player2_input = 0;
	//AI::add_tank(&m_ai, m_playertank, AI_TYPE_EVADER);
	AI::add_tank(&m_ai, m_player2tank, AI_TYPE_ATTACKER);
	AI::add_tank(&m_ai, m_player3tank, AI_TYPE_ATTACKER);
	
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
	m_frames_elapsed = 0;
}

GSGame::~GSGame(){
	BasicTank::destroy(&m_tanks);
	TankBullet::destroy(&m_bullets);
	cudaFree(m_cuda_tanks);
	cudaFree(m_cuda_bullets);
	cudaFree(m_cuda_runner);
	cudaFree(m_cuda_ai);
}

void GSGame::onSceneDeactivate(){
}

void GSGame::onSceneActivate(){
	m_timer.restart();
}

#include <iostream>

using namespace std;

void GSGame::onFrameRender(CL_GraphicContext* gc){
	gc->set_frame_buffer(m_offscreen_buffer);
	
	if(GameDisplay::s_usecuda){
		// re-assign the pointer to the CPU version
		BasicTank::reset_pointers(&m_tanks, m_physrunner.get(), &m_bullets);
		TankBullet::reset_phys_pointer(&m_bullets, m_physrunner.get());
	}
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
	CL_StringFormat fmt("States: %1 %2 %3 %4 | Player: %5 %6 %7 | Time elapsed: %8");
	fmt.set_arg(1, m_ai.bullet_vector[0]);
	fmt.set_arg(2, m_ai.tank_vector[0]);
	fmt.set_arg(3, m_ai.direction_state[0]);
	fmt.set_arg(4, m_ai.distance_state[0]);
	pos = Physics::PhysRunner::get_cur_pos(m_physrunner.get(),
														 m_tanks.phys_id[m_playertank]);
	fmt.set_arg(5, pos.x);
	fmt.set_arg(6, pos.y);
	fmt.set_arg(7, BasicTank::get_tank_rot(&m_tanks, m_playertank));
	fmt.set_arg(8, m_frames_elapsed);
	m_dbgmsg = fmt.get_result();
	m_debugfont->draw_text(*gc, 1, 12, m_dbgmsg, CL_Colorf::red);
	
	gc->reset_frame_buffer();
	gc->set_texture(0, m_offscreen_texture);
	gc->set_program_object(m_shader, cl_program_matrix_modelview_projection);
	m_shader.set_uniform1i("SourceTexture", 1);
	m_shader.set_uniform1f("Amount", 0.25f);
	m_shader.set_uniform1f("Timer", m_timer.elapsed());
	draw_texture(*gc, CL_Rect(0, 0, gc->get_width(), gc->get_height()));
	gc->reset_program_object();
	gc->reset_texture(0);
}

// This function was taken from the ClanLib post-processing example
void GSGame::draw_texture(CL_GraphicContext &gc,
						  const CL_Rectf &rect,
						  const CL_Colorf &color,
						  const CL_Rectf &texture_unit1_coords){
	CL_Vec2f positions[6] =
	{
		CL_Vec2f(rect.left, rect.top),
				 CL_Vec2f(rect.right, rect.top),
				 CL_Vec2f(rect.left, rect.bottom),
				 CL_Vec2f(rect.right, rect.top),
				 CL_Vec2f(rect.left, rect.bottom),
				 CL_Vec2f(rect.right, rect.bottom)
	};
	
	CL_Vec2f tex1_coords[6] =
	{
		CL_Vec2f(texture_unit1_coords.left, texture_unit1_coords.top),
				 CL_Vec2f(texture_unit1_coords.right, texture_unit1_coords.top),
				 CL_Vec2f(texture_unit1_coords.left, texture_unit1_coords.bottom),
				 CL_Vec2f(texture_unit1_coords.right, texture_unit1_coords.top),
				 CL_Vec2f(texture_unit1_coords.left, texture_unit1_coords.bottom),
				 CL_Vec2f(texture_unit1_coords.right, texture_unit1_coords.bottom)
	};
	
	CL_PrimitivesArray prim_array(gc);
	prim_array.set_attributes(0, positions);
	prim_array.set_attribute(1, color);
	prim_array.set_attributes(2, tex1_coords);
	gc.draw_primitives(cl_triangles, 6, prim_array);
}

// #if __CUDA_ARCH__
__global__ void gsgame_step(f32 dt,
							tank_id player_tank,
							Physics::PhysRunner::RunnerCore* runner,
							TankBullet::BulletCollection* bullets,
							BasicTank::TankCollection* tanks,
							AI::AI_Core* aic,
							u32 player_input){
	int idx = threadIdx.x;
	
	// AI operations
	AI::timestep(aic, dt);
	
	if(idx == 0){
		// thread 0 will perform the input processing
		// all other threads will do squat (wasteful but can't be helped)
		if(tanks->state[player_tank] != TANK_STATE_INACTIVE){
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
	}
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

void GSGame::onFrameUpdate(double dt,
						   CL_InputDevice* keyboard,
						   CL_InputDevice* mouse){
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
												   m_cuda_ai,
												   m_player_input);
						  
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
	}else{
		// process the player input
		if(m_tanks.state[m_playertank] != TANK_STATE_INACTIVE){
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
			AI::timestep(&m_ai, dt);
			Physics::PhysRunner::timestep(m_physrunner.get(), dt);
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
	}
	
	// update the sprites
	m_background->update();
	m_testbullet->update();
	m_testtank->update();
	if(m_tanks.state[0] != TANK_STATE_INACTIVE){
		++m_frames_elapsed;
	}
}

