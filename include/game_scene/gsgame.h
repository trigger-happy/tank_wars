/*
    <one line to give the library's name and an idea of what it does.>
    Copyright (C) <year>  <name of author>

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/
#ifndef GSGAME_H
#define GSGAME_H
#include <ClanLib/core.h>
#include <ClanLib/display.h>
#include <boost/scoped_ptr.hpp>

#include "game_scene/igamescene.h"
#include "game_core/tankbullet.h"
#include "game_core/basictank.h"
#include "game_core/tank_ai.h"
#include "types.h"

#define PLAYER_FORWARD	1
#define PLAYER_BACKWARD	2
#define PLAYER_LEFT		4
#define PLAYER_RIGHT	8
#define PLAYER_FIRE		16
#define PLAYER_STOP		32

namespace Physics{
	namespace PhysRunner{
		struct RunnerCore;
	}
}

namespace TankBullet{
	struct BulletCollection;
}

namespace BasicTank{
	struct TankCollection;
}

class GSGame : public iGameScene{
public:
	GSGame(CL_GraphicContext& gc, CL_ResourceManager& resources);
	virtual ~GSGame();
	
    virtual void onSceneDeactivate();
    virtual void onSceneActivate();
    virtual void onFrameRender(CL_GraphicContext* gc);
    virtual void onFrameUpdate(double dt,
							   CL_InputDevice* keyboard,
							   CL_InputDevice* mouse);
private:
	boost::scoped_ptr<Physics::PhysRunner::RunnerCore> m_physrunner;
	boost::scoped_ptr<CL_Sprite> m_background;
	boost::scoped_ptr<CL_Sprite> m_testbullet;
	boost::scoped_ptr<CL_Sprite> m_testtank;
	boost::scoped_ptr<CL_Sprite> m_testtank2;
	boost::scoped_ptr<CL_Font_System> m_debugfont;
	CL_String m_dbgmsg;
	
	
	TankBullet::BulletCollection m_bullets;
	BasicTank::TankCollection m_tanks;
	AI::AI_Core m_ai;
	tank_id m_playertank;
	tank_id m_player2tank;
	u32 m_player_input;
	u32 m_player2_input;
	
	//cuda stuff
	Physics::PhysRunner::RunnerCore* m_cuda_runner;
	TankBullet::BulletCollection* m_cuda_bullets;
	BasicTank::TankCollection* m_cuda_tanks;
	AI::AI_Core* m_cuda_ai;
// 	u8* m_cuda_player_input;
};

#endif // GSGAME_H
