#include <kcpolydb.h>
#include <iostream>
#include <string>
#include <cmath>
#include <boost/scoped_ptr.hpp>
#include "data_store/data_store.h"
#include "data_store/ds_types.h"
#include "evolvers/ievolver.h"

using namespace std;
using namespace kyotocabinet;

bool compare_genes(const ai_data& aid1, const ai_data& aid2){
	// ignore the score, just the genetic data
	bool result = true;
	for(int i = 0; i < MAX_GENE_DATA; ++i){
		for(int j = 0; j < MAX_AI_EVADERS; ++j){
			result &= (aid1.gene_accel[i][j] == aid2.gene_accel[i][j]);
			result &= (aid1.gene_heading[i][j] == aid2.gene_heading[i][j]);
		}
	}
	return result;
}

int main(int argc, char** argv){
	if(argc < 3){
		cerr << "Usage: " << argv[0] << " gene1.kch gene2.kch" << endl;
		return -1;
	}

	boost::scoped_ptr<PolyDB> ds1(new PolyDB());
	boost::scoped_ptr<PolyDB> ds2(new PolyDB());
	bool result = ds1->open(argv[1], PolyDB::OREADER);
	result &= ds2->open(argv[2], PolyDB::OREADER);

	if(result){
		ai_key aik;
		ai_data aid1, aid2;
		for(int i = 0; i < NUM_INSTANCES; ++i){
			for(int j = 1; j <= MAX_GENERATIONS; ++j){
				aik.id = i;
				aik.generation = j;

				result = ds1->get(reinterpret_cast<const char*>(&aik), sizeof(ai_key),
								  reinterpret_cast<char*>(&aid1), sizeof(ai_data));
				if(!result){
					cerr << "Failed to load: ds1 " << i << " " << j << endl;
				}

				result = ds2->get(reinterpret_cast<const char*>(&aik), sizeof(ai_key),
								  reinterpret_cast<char*>(&aid2), sizeof(ai_data));

				if(!result){
					cerr << "Failed to load: ds2 " << i << " " << j << endl;
				}

				result = compare_genes(aid1, aid2);
				if(!result){
					cout << "Genes differ at: " << i << " " << j << endl;
				}
			}
		}
	}else{
		cerr << "Failed to open one or both genetic databases" << endl;
	}

	ds1->close();
	ds2->close();
	
	return 0;
}