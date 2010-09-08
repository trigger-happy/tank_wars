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

#ifndef EVOLVER_CPU_H
#define EVOLVER_CPU_H
#include <string>
#include <vector>
#include <boost/scoped_ptr.hpp>
#include "evolvers/ievolver.h"
#include "game_core/tankbullet.h"
#include "game_core/basictank.h"
#include "game_core/tank_ai.h"

class Evolver_cpu : public iEvolver<Evolver_cpu>{
public:
private:
	friend class iEvolver<Evolver_cpu>;
	void initialize_impl();
	void cleanup_impl();
	void frame_step_impl(float dt);
	void retrieve_state_impl();
	void evolve_ga_impl();
	u32 retrieve_score_impl();
	void save_best_gene_impl(const std::string& fname);
	void prepare_game_state_impl();
	bool is_game_over_impl();

private:
	std::vector<Physics::PhysRunner::RunnerCore> m_physrunner;
	std::vector<TankBullet::BulletCollection> m_bullets;
	std::vector<BasicTank::TankCollection> m_tanks;
	std::vector<AI::AI_Core> m_ai;
	
};

#endif // EVOLVER_CPU_H
