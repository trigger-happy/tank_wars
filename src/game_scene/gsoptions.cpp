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


#include "game_scene/gsoptions.h"

GSOptions::GSOptions(CL_GraphicContext& gc, CL_ResourceManager& resources){
	CL_FontDescription desc;
	desc.set_typeface_name("tahoma");
	desc.set_height(32);
	m_font = new CL_Font_System(gc, desc);
	//TODO: add more code here for initializing the scene
}

GSOptions::~GSOptions(){
	if(m_font){
		delete m_font;
		m_font = NULL;
	}
}

void GSOptions::onFrameUpdate(double dt,
							  CL_InputDevice* keyboard,
							  CL_InputDevice* mouse){
	//TODO: code here to react to events
}

void GSOptions::onFrameRender(CL_GraphicContext* gc){
	//TODO: code here for rendering
}

void GSOptions::onSceneActivate(){
}

void GSOptions::onSceneDeactivate(){
}