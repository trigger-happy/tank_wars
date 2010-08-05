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
#ifndef GAME_DISPLAY_H
#define GAME_DISPLAY_H
#include <boost/timer.hpp>
#include <stack>

class iGameScene;

class GameDisplay{
public:
	static int main();
	
	static void push_scene(iGameScene* scene);
	static void pop_scene();
	
	
	static bool s_running;
	
private:
	static boost::timer s_frame_timer;
	static double s_deltatime;
	static std::stack<iGameScene*> s_scene_stack;
};

#endif //GAME_DISPLAY_H