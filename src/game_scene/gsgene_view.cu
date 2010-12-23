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
#include <iostream>
#include <cuda.h>
#include "game_display.h"
#include "data_store/data_store.h"
#include "data_store/ds_types.h"
#include "game_scene/gsgene_view.h"
#include "../game_core/twgame.cu"
// #include "game_core/physics.h"
// #include "game_core/tankbullet.h"
// #include "game_core/basictank.h"

#define SCALE_FACTOR 12

#define CUDA_BLOCKS 1
#define CUDA_THREADS MAX_ARRAY_SIZE

using namespace std;

extern DataStore* g_db;
extern ai_key g_aik;

// helper function
void apply_transform(CL_GraphicContext* gc, Physics::vec2& c){
	c.x *= SCALE_FACTOR;
	c.y *= -SCALE_FACTOR;
	c.x += gc->get_width()/2;
	c.y += gc->get_height()/2;
}

GSGeneView::GSGeneView(CL_GraphicContext& gc, CL_ResourceManager& resources)
: m_physrunner(new Physics::PhysRunner::RunnerCore()){

	// quick output of the format of things
// 	cout << "bullet_vector tank_vector direction_state distance_state | tank.x tank.y tank.rot | bullet.x bullet.y bullet.rot" << endl;
	
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
	u32 score = 0;
	// set the gene data
	g_db->get_gene_data(g_aik, score, this->get_ai());

	m_ai_b = m_ai;
	
	// test code
// 	Physics::vec2 params;
// 	params.x = 0;
// 	m_playertank = BasicTank::spawn_tank(&m_tanks, params, 90, 0);
// 	params.x = 0;
// 	params.y = 12;
// 	m_player2tank = BasicTank::spawn_tank(&m_tanks, params, 180, 1);
// 	AI::add_tank(&m_ai, m_playertank, AI_TYPE_EVADER);
// 	AI::add_tank(&m_ai, m_player2tank, AI_TYPE_ATTACKER);

	// prepare the initial scenario
	m_test_dist = 1;
	m_test_sect = 0;
	m_test_vect = 0;
	prepare_game_scenario(m_test_dist, m_test_sect, m_test_vect);

	// load a saved gene for viewing
	if(GameDisplay::s_view_gene){
		std::ifstream fin("report.dat");
		AI::AI_Core::gene_type tempval;
		for(int i = 0; i < MAX_GENE_DATA; ++i){
			fin.read((char*)&tempval, sizeof(tempval));
			m_ai.gene_accel[i][0] = tempval;
		}
		for(int i = 0; i < MAX_GENE_DATA; ++i){
			fin.read((char*)&tempval, sizeof(tempval));
			m_ai.gene_heading[i][0] = tempval;
		}
	}
	
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
	m_runningscore = 0;
}

void GSGeneView::prepare_game_scenario(u32 dist, u32 bullet_loc, u32 bullet_vec){
	Physics::PhysRunner::initialize(m_physrunner.get());
	TankBullet::initialize(&m_bullets, m_physrunner.get());
	BasicTank::initialize(&m_tanks, m_physrunner.get(), &m_bullets);
	AI::initialize(&m_ai, &m_tanks, &m_bullets);

	m_ai = m_ai_b;

	Physics::vec2 atk_params;
	
	f32 theta = SECTOR_SIZE*bullet_loc + SECTOR_SIZE/2;
	f32 hypot = DISTANCE_FACTOR*dist + DISTANCE_FACTOR/2;
	atk_params.y = hypot * sin(util::degs_to_rads(theta));
	atk_params.x = hypot * cos(util::degs_to_rads(theta));
	f32 tank_rot = VECTOR_SIZE*bullet_vec + VECTOR_SIZE/2;
	
	Physics::vec2 params;
	params.x = 0;
	m_playertank = BasicTank::spawn_tank(&m_tanks,
													params,
													90,
													0);
	AI::add_tank(&m_ai, m_playertank, AI_TYPE_EVADER);

	m_player2tank = BasicTank::spawn_tank(&m_tanks,
													atk_params,
													tank_rot,
													1);
	AI::add_tank(&m_ai, m_player2tank, AI_TYPE_ATTACKER);
}

GSGeneView::~GSGeneView(){
	BasicTank::destroy(&m_tanks);
	TankBullet::destroy(&m_bullets);
	cudaFree(m_cuda_tanks);
	cudaFree(m_cuda_bullets);
	cudaFree(m_cuda_runner);
	cudaFree(m_cuda_ai);
}

void GSGeneView::onSceneDeactivate(){
}

void GSGeneView::onSceneActivate(){
	m_timer.restart();
}

#include <iostream>

using namespace std;

void GSGeneView::onFrameRender(CL_GraphicContext* gc){
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
	
// 	if(m_tanks.state[m_player3tank] != TANK_STATE_INACTIVE){
// 		pos = BasicTank::get_tank_pos(&m_tanks, m_player3tank);
// 		apply_transform(gc, pos);
// 		f32 rot = BasicTank::get_tank_rot(&m_tanks, m_player3tank);
// 		m_testtank2->set_angle(CL_Angle(-rot, cl_degrees));
// 		m_testtank2->draw(*gc, pos.x, pos.y);
// 	}
	
	// Debug info
	CL_StringFormat fmt("States: %1 %2 %3 %4 | Player pos: %5 %6 | Running Score: %7");
	fmt.set_arg(1, m_ai.bullet_vector[0]);
	fmt.set_arg(2, m_ai.tank_vector[0]);
	fmt.set_arg(3, m_ai.direction_state[0]);
	fmt.set_arg(4, m_ai.distance_state[0]);
	pos = Physics::PhysRunner::get_cur_pos(m_physrunner.get(),
														 m_tanks.phys_id[m_playertank]);
	fmt.set_arg(5, pos.x);
	fmt.set_arg(6, pos.y);
	fmt.set_arg(7, m_runningscore);
	m_dbgmsg = fmt.get_result();
	m_debugfont->draw_text(*gc, 1, 12, m_dbgmsg, CL_Colorf::red);
}


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

void GSGeneView::onFrameUpdate(double dt,
						   CL_InputDevice* keyboard,
						   CL_InputDevice* mouse){
	// keyboard control processing
	// test code for the time being
	
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
	}else{
		// process the player input
		if(m_tanks.state[m_playertank] != TANK_STATE_INACTIVE
			&& m_tanks.state[m_player2tank] != TANK_STATE_INACTIVE){
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
		}else if(m_tanks.state[m_playertank] == TANK_STATE_INACTIVE){
			// player tank died, let's reset
			++m_test_vect;
			if(m_test_vect >= 1){
				m_test_vect = 0;
				++m_test_sect;
				if(m_test_sect >= NUM_LOCATION_STATES){
					m_test_sect = 0;
					++m_test_dist;
					if(m_test_dist >= NUM_DISTANCE_STATES){
						cout << m_runningscore+1 << endl;
						GameDisplay::s_running = false;
					}
				}
			}
			prepare_game_scenario(m_test_dist, m_test_sect, m_test_vect);
		}else if(m_tanks.state[m_player2tank] == TANK_STATE_INACTIVE){
			++m_runningscore;
			++m_test_vect;
			if(m_test_vect >= 1){
				m_test_vect = 0;
				++m_test_sect;
				if(m_test_sect >= NUM_LOCATION_STATES){
					m_test_sect = 0;
					++m_test_dist;
					if(m_test_dist >= NUM_DISTANCE_STATES){
						cout << m_runningscore+1 << endl;
						GameDisplay::s_running = false;
					}
				}
			}
			prepare_game_scenario(m_test_dist, m_test_sect, m_test_vect);
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

