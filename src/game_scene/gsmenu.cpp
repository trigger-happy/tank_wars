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

#include <cassert>
#include <ClanLib/core.h>
#include <ClanLib/display.h>
#include <ClanLib/gl.h>
#include "game_scene/gsmenu.h"

extern bool s_running;

GSMenu::GSMenu(CL_GraphicContext& gc, CL_ResourceManager& resources)
: m_titlesprite(gc, "main_menu/title_sprite", &resources){
	m_playgame_btn.initialize(gc,
							  resources,
							  "main_menu/playgame_btn",
							  CL_Vec2<float>(400, 200));
							  
	m_option_btn.initialize(gc,
							resources,
							"main_menu/option_btn",
							CL_Vec2<float>(400, 300));
							
	m_quit_btn.initialize(gc,
						  resources,
						  "main_menu/quit_btn",
						  CL_Vec2<float>(400, 400));
}

void GSMenu::onFrameRender(CL_GraphicContext* gc){
	m_titlesprite.draw(*gc, 400, 100);
	
	m_playgame_btn.render(*gc);
	m_option_btn.render(*gc);
	m_quit_btn.render(*gc);
}

void GSMenu::onFrameUpdate(double dt,
						   CL_InputDevice* keyboard,
						   CL_InputDevice* mouse){
	m_titlesprite.update();
	
	switch(m_playgame_btn.mouse_check(*mouse)){
		case 0:
			//TODO: do nothing
			break;
		case 1:
			//TODO: code here for starting the game
			break;
		case 2:
			// do nothing
			break;
		default:
			assert(false && "Bad return code");
	}
	
	switch(m_option_btn.mouse_check(*mouse)){
		case 0:
			//TODO: do nothing
			break;
		case 1:
			//TODO: code here for showing options
			break;
		case 2:
			// do nothing
			break;
		default:
			assert(false && "Bad return code");
	}
	
	switch(m_quit_btn.mouse_check(*mouse)){
		case 0:
			//TODO: do nothing
			break;
		case 1:
			// TODO: code here for quitting
			s_running = false;
			break;
		case 2:
			// do nothing
			break;
		default:
			assert(false && "Bad return code");
	}
	
	m_playgame_btn.frame_update(0);
	m_option_btn.frame_update(0);
	m_quit_btn.frame_update(0);
}

