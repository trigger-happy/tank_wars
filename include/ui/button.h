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

#ifndef BUTTON_H
#define BUTTON_H
#include <ClanLib/core.h>
#include <ClanLib/display.h>

class Button{
public:
	Button(){}
	
	~Button();
	
	void initialize(CL_GraphicContext& gc,
		   CL_ResourceManager& resources,
		   const CL_String& name,
		   const CL_Vec2<float> pos);
		   
	void cleanup();
		   
	void render(CL_GraphicContext& gc);
	
	int mouse_check(CL_InputDevice& mouse);
	
	void frame_update(double dt);
private:
	CL_Sprite* m_sprite;
	CL_Vec2<float> m_pos;
};

#endif // BUTTON_H
