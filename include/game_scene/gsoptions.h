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


#ifndef GSOPTIONS_H
#define GSOPTIONS_H
#include <ClanLib/core.h>
#include <ClanLib/display.h>
#include <boost/scoped_ptr.hpp>
#include "game_scene/igamescene.h"
#include "ui/button.h"

class GSOptions : public iGameScene{
public:
	GSOptions(CL_GraphicContext& gc, CL_ResourceManager& resources);
	virtual ~GSOptions();
	
	virtual void onFrameUpdate(double dt,
							   CL_InputDevice* keyboard,
							   CL_InputDevice* mouse);
	virtual void onFrameRender(CL_GraphicContext* gc);
	virtual void onSceneActivate();
	virtual void onSceneDeactivate();
	
private:
	boost::scoped_ptr<CL_Font_System> m_font;
	Button m_backbtn;
	//TODO: add more stuff here for the options screen
};

#endif // GSOPTIONS_H
