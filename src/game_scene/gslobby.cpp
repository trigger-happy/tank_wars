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

#include "game_scene/gslobby.h"

GSLobby::GSLobby(CL_GraphicContext& gc, CL_ResourceManager& resources){
	CL_FontDescription desc;
	desc.set_typeface_name("tahoma");
	desc.set_height(32);
	m_font = new CL_Font_System(gc, desc);
	m_xpos = gc.get_width()/2-100;
	m_ypos = gc.get_height()/4;
}

GSLobby::~GSLobby(){
	if(m_font){
		delete m_font;
		m_font = NULL;
	}
}

void GSLobby::onFrameUpdate(double dt,
						   CL_InputDevice* keyboard,
						   CL_InputDevice* mouse){
}
						   
void GSLobby::onFrameRender(CL_GraphicContext* gc){
	m_font->draw_text(*gc, m_xpos, m_ypos, "Nothing here yet");
}

void GSLobby::onSceneActivate(){
}

void GSLobby::onSceneDeactivate(){
}