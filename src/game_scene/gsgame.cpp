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
#include "game_scene/gsgame.h"
#include "game_core/physics.h"
#include "game_core/tankbullet.h"
#include "game_core/basictank.h"

#define SCALE_FACTOR 12

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
}

GSGame::~GSGame(){
	m_tanks.destroy();
	m_bullets.destroy();
}

void GSGame::onSceneDeactivate(){
}

void GSGame::onSceneActivate(){
}

void GSGame::onFrameRender(CL_GraphicContext* gc){
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

void GSGame::onFrameUpdate(double dt,
						   CL_InputDevice* keyboard,
						   CL_InputDevice* mouse){
	//TODO: code here for AI update
	
	// keyboard control processing
	// test code for the time being
	if(keyboard->get_keycode(CL_KEY_SPACE)){
		m_tanks.fire(m_playertank);
	}
	
	if(keyboard->get_keycode(CL_KEY_RETURN)){
		m_tanks.stop(m_playertank);
	}
	
	if(keyboard->get_keycode(CL_KEY_UP)){
		m_tanks.move_forward(m_playertank);
	}else if(keyboard->get_keycode(CL_KEY_DOWN)){
		m_tanks.move_backward(m_playertank);
	}
	
	if(keyboard->get_keycode(CL_KEY_LEFT)){
		m_tanks.turn_left(m_playertank);
	}else if(keyboard->get_keycode(CL_KEY_RIGHT)){
		m_tanks.turn_right(m_playertank);
	}
	
	// update the game state
	//TODO: add support for running on the gpu when possible.
	m_physrunner->timestep(dt);
	m_bullets.update(dt);
	m_tanks.update(dt);
	
	// update the sprites
	m_background->update();
	m_testbullet->update();
	m_testtank->update();
}

