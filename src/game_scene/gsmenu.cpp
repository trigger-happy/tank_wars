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

#include <ClanLib/core.h>
#include <ClanLib/display.h>
#include <ClanLib/gl.h>
#include "game_scene/gsmenu.h"

GSMenu::GSMenu(CL_GraphicContext& gc, CL_ResourceManager& resources)
: m_titlesprite(gc, "main_menu/title_sprite", &resources){
}

void GSMenu::onFrameRender(CL_GraphicContext* gc){
	m_titlesprite.draw(*gc, 400, 100);
}

void GSMenu::onFrameUpdate(double dt,
						   CL_InputDevice* keyboard,
						   CL_InputDevice* mouse){
	m_titlesprite.update();
}

