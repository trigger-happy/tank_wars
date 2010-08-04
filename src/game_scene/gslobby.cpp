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
#include "game_display.h"
#include "game_scene/gslobby.h"
#include "game_scene/gsgame.h"

GSLobby::GSLobby(CL_GraphicContext& gc, CL_ResourceManager& resources)
: m_gsgame(new GSGame(gc, resources)){
	CL_FontDescription desc;
	desc.set_typeface_name("tahoma");
	desc.set_height(32);
	m_font.reset(new CL_Font_System(gc, desc));
	m_xpos = gc.get_width()/2-100;
	m_ypos = gc.get_height()/4;
	m_playbtn.initialize(gc, resources, "game_lobby/play_btn",
						 CL_Vec2<float>(m_xpos, m_ypos+100));
	m_backbtn.initialize(gc, resources, "game_lobby/back_btn",
						 CL_Vec2<float>(m_xpos, m_ypos+200));
}

GSLobby::~GSLobby(){
}

void GSLobby::onFrameUpdate(double dt,
						   CL_InputDevice* keyboard,
						   CL_InputDevice* mouse){
	switch(m_playbtn.mouse_check(*mouse)){
		case 0:
			// do nothing
			break;
		case 1:
			//TODO: start the game
			break;
		case 2:
			// do nothing
			break;
		default:
			assert(false && "Bad return code");
	}
	
	switch(m_backbtn.mouse_check(*mouse)){
		case 0:
			// do nothing
			break;
		case 1:
			// go back
			GameDisplay::pop_scene();
			break;
		case 2:
			// do nothing
			break;
		default:
			assert(false && "Bad return code");
	}
	m_playbtn.frame_update(0);
	m_backbtn.frame_update(0);
}
						   
void GSLobby::onFrameRender(CL_GraphicContext* gc){
	m_font->draw_text(*gc, m_xpos, m_ypos, "Nothing here yet");
	m_playbtn.render(*gc);
	m_backbtn.render(*gc);
}

void GSLobby::onSceneActivate(){
}

void GSLobby::onSceneDeactivate(){
}