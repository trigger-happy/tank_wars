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

#ifndef EVOLVER_GPU_H
#define EVOLVER_GPU_H
#include "evolvers/ievolver.h"

class Evolver_gpu : public iEvolver<Evolver_gpu>{
public:
private:
	friend class iEvolver<Evolver_gpu>;
	void initialize_impl();
	void cleanup_impl();
	void frame_step_impl(float dt);
	void retrieve_state_impl();
};

#endif // EVOLVER_GPU_H
