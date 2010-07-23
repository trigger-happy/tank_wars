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

#include "ui/button.h"


Button::~Button(){
	cleanup();
}

void Button::initialize(CL_GraphicContext& gc,
			   CL_ResourceManager& resources,
			   const CL_String& name,
			   const CL_Vec2<float> pos){
	m_sprite = new CL_Sprite(gc, name, &resources);
	m_pos = pos;
}

void Button::cleanup(){
	if(m_sprite){
		delete m_sprite;
	}
}
	   
void Button::render(CL_GraphicContext& gc){
	m_sprite->draw(gc, m_pos.x, m_pos.y);
}

int Button::mouse_check(CL_InputDevice& mouse){
	CL_Vec2<float> pos(mouse.get_x(), mouse.get_y());
	CL_Size s = m_sprite->get_size();
	if(pos.x >= (m_pos.x - s.width/2)
		&& pos.x <= (m_pos.x + s.width/2)
		&& pos.y >= m_pos.y - (s.height/2)
		&& pos.y <= m_pos.y + (s.height/2)){
		
		// it's inside the hitbox
		if(mouse.get_keycode(CL_MOUSE_LEFT)){
			// the user clicked in here
			m_sprite->set_frame(1);
			return 1;
		}else{
			// the mouse is just hovering here
			m_sprite->set_frame(2);
			return 2;
		}
	}else{
		m_sprite->set_frame(0);
		return 0;
	}
}

void Button::frame_update(double dt){
	m_sprite->update(dt);
}