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

#ifndef GSLOBBY_H
#define GSLOBBY_H
#include <ClanLib/core.h>
#include <ClanLib/display.h>
#include "igamescene.h"

class GSLobby : public iGameScene{
public:
	GSLobby(CL_GraphicContext& gc, CL_ResourceManager& resources);
	virtual ~GSLobby();
	
	virtual void onFrameUpdate(double dt,
							   CL_InputDevice* keyboard,
							   CL_InputDevice* mouse);
	virtual void onFrameRender(CL_GraphicContext* gc);
	virtual void onSceneActivate();
	virtual void onSceneDeactivate();
	
private:
	CL_Font_System* m_font;
	int m_xpos;
	int m_ypos;
};

#endif // GSLOBBY_H
