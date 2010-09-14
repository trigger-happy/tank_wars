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
							   const AI::AI_Core& aic){
	bool result = true;
	ai_data aid;

	// copy the gene from aic to aid
	for(u32 i = 0; i < MAX_GENE_DATA; ++i){
		for(u32 j = 0; j < MAX_AI_EVADERS; ++j){
			aid.gene_accel[i][j] = aic.gene_accel[i][j];
			aid.gene_heading[i][j] = aic.gene_heading[i][j];
		}
	}

	// save the data
	result = m_aidb->set(reinterpret_cast<const char*>(&key), sizeof(ai_key),
					  reinterpret_cast<const char*>(&aid), sizeof(ai_data));
	return result;
}

bool DataStore::get_gene_data(const ai_key& key,
							  AI::AI_Core& aic){
	bool result = true;
	ai_data aid;

	result = m_aidb->get(reinterpret_cast<const char*>(&key), sizeof(ai_key),
					  reinterpret_cast<char*>(&aid), sizeof(ai_data));

	// copy the gene from aid to aic
	for(u32 i = 0; i < MAX_GENE_DATA; ++i){
		for(u32 j = 0; j < MAX_AI_EVADERS; ++j){
			aic.gene_accel[i][j] = aid.gene_accel[i][j];
			aid.gene_heading[i][j] = aid.gene_heading[i][j];
		}
	}
	
	return result;
}

const char* DataStore::get_ai_error() const{
	return m_aidb->error().name();
}

const char* DataStore::get_sim_error() const{
	return m_simdb->error().name();
}