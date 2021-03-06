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
// #include <boost/format.hpp>
#include "game_display.h"
#include "game_scene/gssim_view.h"
#include "../game_core/twgame.cu"
// #include "game_core/physics.h"
// #include "game_core/tankbullet.h"
// #include "game_core/basictank.h"

#define SCALE_FACTOR 12

#define CUDA_BLOCKS 1
#define CUDA_THREADS MAX_ARRAY_SIZE

using namespace std;
// using namespace boost;

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

	m_playertank = 0;
	m_player2tank = 1;

	// get the simulation data
	m_tanks = m_simd.tc;
	m_bullets = m_simd.bc;
	m_physrunner->bodies = m_simd.bodies[0];

	// reset the pointers
	BasicTank::reset_pointers(&m_tanks, m_physrunner.get(), &m_bullets);
	TankBullet::reset_phys_pointer(&m_bullets, m_physrunner.get());

	m_frames_elapsed = 0;
}

GSSimView::~GSSimView(){
	BasicTank::destroy(&m_tanks);
	TankBullet::destroy(&m_bullets);
}

void GSSimView::onSceneDeactivate(){
}

void GSSimView::onSceneActivate(){
	m_timer.restart();
}

#include <iostream>

using namespace std;

void GSSimView::onFrameRender(CL_GraphicContext* gc){	
	// draw the background
	Physics::vec2 pos;
	apply_transform(gc, pos);
	m_background->draw(*gc, (f32)pos.x, (f32)pos.y);
	
	// draw the bullets, yes, we're cheating the numbers
	// OOP can wait another day
	for(int i = 0; i < 3; ++i){
		pos = TankBullet::get_bullet_pos(&m_bullets, i);
		apply_transform(gc, pos);
// 		if(m_bullets.state[i] != BULLET_STATE_INACTIVE){
			m_testbullet->draw(*gc, (f32)pos.x, (f32)pos.y);
// 		}
	}
	
	// draw the tanks
// 	if(m_tanks.state[m_playertank] != TANK_STATE_INACTIVE){
		pos = BasicTank::get_tank_pos(&m_tanks, m_playertank);
		apply_transform(gc, pos);
		f32 rot = BasicTank::get_tank_rot(&m_tanks, m_playertank);
		m_testtank->set_angle(CL_Angle(-rot, cl_degrees));
		m_testtank->draw(*gc, (f32)pos.x, (f32)pos.y);
// 	}

// 	if(m_tanks.state[m_player2tank] != TANK_STATE_INACTIVE){
		pos = BasicTank::get_tank_pos(&m_tanks, m_player2tank);
		apply_transform(gc, pos);
		rot = BasicTank::get_tank_rot(&m_tanks, m_player2tank);
		m_testtank2->set_angle(CL_Angle(-rot, cl_degrees));
		m_testtank2->draw(*gc, (f32)pos.x, (f32)pos.y);
// 	}
	
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

void GSSimView::onFrameUpdate(double dt,
						   CL_InputDevice* keyboard,
						   CL_InputDevice* mouse){
	if(m_frames_elapsed < MAX_BODY_RECORD){

		// perform all the update
		//AI::timestep(&m_ai, dt);
		//Physics::PhysRunner::timestep(m_physrunner.get(), dt);
		// copy over the timestep data
		m_physrunner->bodies = m_simd.bodies[m_frames_elapsed];
		Physics::vec2 pos = m_physrunner->bodies.cur_pos.get_vec2(m_tanks.phys_id[0]);
		f32 rot = m_physrunner->bodies.rotation[m_tanks.phys_id[0]];
// 		cout << format("E.x: %1 E.y: %2 E.r: %3") % pos.x % pos.y % rot << endl;
// 		printf("E.x: %.12f E.y: %.12f E.r: %.12f\n", pos.x, pos.y, rot);

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
	
	// update the sprites
	m_background->update();
	m_testbullet->update();
	m_testtank->update();
// 	if(m_tanks.state[0] != TANK_STATE_INACTIVE){
	Physics::vec2 pos = BasicTank::get_tank_pos(&m_tanks, m_playertank);
// 	if(pos.x != OFFSCREEN_X && pos.y != OFFSCREEN_Y){
		++m_frames_elapsed;
// 	}
}

