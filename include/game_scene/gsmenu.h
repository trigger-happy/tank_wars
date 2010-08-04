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

#ifndef GSMENU_H
#define GSMENU_H
#include <ClanLib/core.h>
#include <ClanLib/display.h>
#include <boost/scoped_ptr.hpp>
#include "game_scene/igamescene.h"
#include "ui/button.h"

class GSOptions;
class GSLobby;

class GSMenu : public iGameScene{
public:
	GSMenu(CL_GraphicContext& gc, CL_ResourceManager& resources);
	virtual ~GSMenu();
	
    virtual void onFrameRender(CL_GraphicContext* gc);
    virtual void onFrameUpdate(double dt,
							   CL_InputDevice* keyboard,
							   CL_InputDevice* mouse);
	virtual void onSceneActivate(){}
	virtual void onSceneDeactivate(){}

private:
	CL_Sprite m_titlesprite;
	Button m_playgame_btn;
	Button m_option_btn;
	Button m_quit_btn;
	
	boost::scoped_ptr<GSLobby> m_gslobby;
	boost::scoped_ptr<GSOptions> m_gsoptions;
};

#endif // GSMENU_H
