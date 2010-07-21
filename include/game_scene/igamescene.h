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

#ifndef IGAMESCENE_H
#define IGAMESCENE_H

class CL_InputDevice;
class CL_GraphicContext;

class iGameScene{
public:
	/*!
	Function called before rendering
	\param dt - The frame time in milliseconds
	\param keyboard - Pointer to a CL_InputDevice for the keyboard
	\param mouse - Pointer to a CL_InputDevice for the mouse
	*/
	virtual void onFrameUpdate(double dt,
							   CL_InputDevice* keyboard,
							   CL_InputDevice* mouse) = 0;
							  
	/*!
	Function called when rendering on the screen
	\param gc - Pointer to the CL_GraphicContext to draw on
	*/
	virtual void onFrameRender(CL_GraphicContext* gc) = 0;
};

#endif //IGAMESCENE_H