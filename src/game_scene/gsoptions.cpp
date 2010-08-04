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
#include "game_scene/gsoptions.h"
#include "game_display.h"

GSOptions::GSOptions(CL_GraphicContext& gc, CL_ResourceManager& resources){
	CL_FontDescription desc;
	desc.set_typeface_name("tahoma");
	desc.set_height(32);
	m_font.reset(new CL_Font_System(gc, desc));
	m_backbtn.initialize(gc, resources, "game_lobby/back_btn",
						 CL_Vec2f(gc.get_width()/2,
								  gc.get_height()-gc.get_height()/4));
	//TODO: add more code here for initializing the scene
}

GSOptions::~GSOptions(){
}

void GSOptions::onFrameUpdate(double dt,
							  CL_InputDevice* keyboard,
							  CL_InputDevice* mouse){
	//TODO: code here to react to events
	switch(m_backbtn.mouse_check(*mouse)){
		case 0:
			// do nothing
			break;
		case 1:
			// go back to the main menu
			GameDisplay::pop_scene();
			break;
		case 2:
			// do nothing
			break;
		default:
			assert(false && "Bad return code");
	}
	m_backbtn.frame_update(0);
}

void GSOptions::onFrameRender(CL_GraphicContext* gc){
	m_font->draw_text(*gc, gc->get_width()/2-50,
					  gc->get_height()/4, "Options");
	m_backbtn.render(*gc);
}

void GSOptions::onSceneActivate(){
}

void GSOptions::onSceneDeactivate(){
}