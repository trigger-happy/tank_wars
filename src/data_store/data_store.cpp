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
#include <kcpolydb.h>
#include <algorithm>
#include <assert.h>
#include "data_store/data_store.h"

using namespace std;
using namespace kyotocabinet;

DataStore::DataStore(const std::string& aidb, const std::string& simdb){
	m_aidb = new PolyDB();
	m_simdb = new PolyDB();
	m_status = m_aidb->open(aidb, PolyDB::OWRITER | PolyDB::OREADER | PolyDB::OCREATE);
	m_status &= m_simdb->open(simdb, PolyDB::OWRITER | PolyDB::OREADER | PolyDB::OCREATE);
}

DataStore::~DataStore(){
	m_aidb->close();
	m_simdb->close();
	delete m_aidb;
	delete m_simdb;
}

bool DataStore::save_gene_data(const ai_key& key,
							   u32 score,
							   const AI::AI_Core& aic,
							   const std::vector<u32>& scenario_results){
	bool result = true;
	ai_data aid;
	aid.score = score;

	// copy the gene from aic to aid
	for(u32 i = 0; i < MAX_GENE_DATA; ++i){
		for(u32 j = 0; j < MAX_AI_EVADERS; ++j){
			aid.gene_accel[i][j] = aic.gene_accel[i][j];
			aid.gene_heading[i][j] = aic.gene_heading[i][j];
		}
	}

// 	copy(scenario_results.begin(), scenario_results.end(), aid.scenario_result);

	s32 accum_score = 0;
	for(int i = 0; i < NUM_SCENARIOS; ++i){
		aid.scenario_result[i] = scenario_results[i];
		accum_score += scenario_results[i];
	}

	assert(accum_score == aid.score);

	// save the data
	result = m_aidb->set(reinterpret_cast<const char*>(&key), sizeof(ai_key),
					  reinterpret_cast<const char*>(&aid), sizeof(ai_data));
	return result;
}

bool DataStore::get_gene_data(const ai_key& key,
							  u32& score,
							  AI::AI_Core& aic,
							  std::vector<u32>& scenario_results){
	bool result = true;
	ai_data aid;

	result = m_aidb->get(reinterpret_cast<const char*>(&key), sizeof(ai_key),
					  reinterpret_cast<char*>(&aid), sizeof(ai_data));

	// copy the gene from aid to aic
	for(u32 i = 0; i < MAX_GENE_DATA; ++i){
		for(u32 j = 0; j < MAX_AI_EVADERS; ++j){
			aic.gene_accel[i][j] = aid.gene_accel[i][j];
			aic.gene_heading[i][j] = aid.gene_heading[i][j];
		}
	}
	score = aid.score;

	scenario_results.clear();
	scenario_results.resize(NUM_SCENARIOS, 0);
// 	copy(aid.scenario_result, aid.scenario_result+NUM_SCENARIOS,
// 		 scenario_results.begin());
	s32 accum_score = 0;
	for(int i = 0; i < NUM_SCENARIOS; ++i){
		scenario_results[i] = aid.scenario_result[i];
		accum_score += aid.scenario_result[i];
	}

	assert(accum_score == aid.score);
	
	return result;
}

const char* DataStore::get_ai_error() const{
	return m_aidb->error().name();
}

const char* DataStore::get_sim_error() const{
	return m_simdb->error().name();
}

bool DataStore::save_sim_data(const sim_key& sk,
							  const sim_data& sd){
	bool result = true;
	result = m_simdb->set(reinterpret_cast<const char*>(&sk), sizeof(sim_key),
						  reinterpret_cast<const char*>(&sd), sizeof(sim_data));
	return result;
}

bool DataStore::get_sim_data(const sim_key& sk,
							 sim_data& sd){
	bool result = true;
	result = m_simdb->get(reinterpret_cast<const char*>(&sk), sizeof(sim_key),
						  reinterpret_cast<char*>(&sd), sizeof(sim_data));
	return result;
}