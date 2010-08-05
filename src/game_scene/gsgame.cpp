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

using namespace Physics;

GSGame::GSGame(CL_GraphicContext& gc, CL_ResourceManager& resources)
: m_physrunner(new PhysRunner()){
	m_background.reset(new CL_Sprite(gc,
									 "game_assets/background",
									 &resources));
}

GSGame::~GSGame(){
}

void GSGame::onSceneDeactivate(){

}

void GSGame::onSceneActivate(){

}

void GSGame::onFrameRender(CL_GraphicContext* gc){
	//TODO: code here for rendering the entire scene
	m_background->draw(*gc, gc->get_width()/2, gc->get_height()/2);
}

void GSGame::onFrameUpdate(double dt,
						   CL_InputDevice* keyboard,
						   CL_InputDevice* mouse){
	//TODO: code here for AI update
	
	//TODO: code here for handling the keyboard controls
	
	// update the game state
	//TODO: add support for running on the gpu when possible.
	m_physrunner->timestep(dt);
	
	// update the sprites
	m_background->update();
}

