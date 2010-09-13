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
#include <map>
#include <utility>
#include <boost/thread.hpp>
#include <boost/array.hpp>
#include "evolvers/ievolver.h"
#include "game_core/tankbullet.h"
#include "game_core/basictank.h"
#include "game_core/tank_ai.h"

#define MAX_THREADS		8

struct frame_updater{
	Physics::PhysRunner::RunnerCore* runners;
	TankBullet::BulletCollection* bullets;
	BasicTank::TankCollection* tanks;
	AI::AI_Core* aic;
	u32* scores;
	u32 start_index;
	u32 end_index;
	u32 dt;

	void operator()();
};

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
	typedef std::map<u32, u32> score_map;
	// genetic stuff
	score_map m_population_score;
	// for debugging purposes
	std::vector<std::pair<u32, u32> > m_last_score;
	std::vector<u32> m_score;

	// frame counter
	u32 m_framecount;
	
	// CPU stuff
	std::vector<Physics::PhysRunner::RunnerCore> m_runner;
	std::vector<TankBullet::BulletCollection> m_bullets;
	std::vector<BasicTank::TankCollection> m_tanks;
	std::vector<AI::AI_Core> m_ai;
	
	// backup of initial data for clean slate
	std::vector<Physics::PhysRunner::RunnerCore> m_runner_b;
	std::vector<TankBullet::BulletCollection> m_bullets_b;
	std::vector<BasicTank::TankCollection> m_tanks_b;
	std::vector<AI::AI_Core> m_ai_b;
	
	// threads
	boost::array<boost::thread, MAX_THREADS> m_threads;
	boost::array<frame_updater, MAX_THREADS> m_updaters;
};

#endif // EVOLVER_CPU_H
