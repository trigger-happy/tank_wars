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

#define SCALE_FACTOR 12

using namespace Physics;

// helper function
void apply_transform(CL_GraphicContext* gc, vec2& c){
	c.x *= SCALE_FACTOR;
	c.y *= SCALE_FACTOR;
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
	m_bullets.initialize(m_physrunner.get());
}

GSGame::~GSGame(){
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
	
	//TODO: draw the test bullet
	pos = m_bullets.get_bullet_pos(0);
	apply_transform(gc, pos);
	m_testbullet->draw(*gc, pos.x, pos.y);
}

void GSGame::onFrameUpdate(double dt,
						   CL_InputDevice* keyboard,
						   CL_InputDevice* mouse){
	//TODO: code here for AI update
	
	//TODO: code here for handling the keyboard controls
	// test code for the time being
	if(keyboard->get_keycode(CL_KEY_SPACE)){
		vec2 pos;
		pos.x = 0;
		pos.y = 0;
		m_bullets.fire_bullet(0, 0, pos);
	}
	
	// update the game state
	//TODO: add support for running on the gpu when possible.
	m_physrunner->timestep(dt);
	m_bullets.update(dt);
	
	// update the sprites
	m_background->update();
}

