#include <kcpolydb.h>
#include <iostream>
#include <string>
#include <cmath>
#include <boost/scoped_ptr.hpp>
#include <boost/program_options.hpp>
#include "data_store/data_store.h"
#include "data_store/ds_types.h"
#include "evolvers/ievolver.h"
#include <boost/concept_check.hpp>

using namespace std;
using namespace kyotocabinet;
namespace po = boost::program_options;

void perform_query(const string& db, u32 id, u32 gen,
					u32 dist, u32 sect, u32 vect){
	boost::scoped_ptr<PolyDB> ds(new PolyDB());
	bool result = ds->open(db, PolyDB::OREADER);

	if(!result){
		cerr << "Failed to open DB" << endl;
		return;
	}

	ai_key aik;
	ai_data aid;

	aik.id = id;
	aik.generation = gen;

	result = ds->get(reinterpret_cast<const char*>(&aik), sizeof(ai_key),
					 reinterpret_cast<char*>(&aid), sizeof(ai_data));

	if(!result){
		cerr << "Failed to query data" << endl;
		return;
	}

	u32 index = (dist * NUM_LOCATION_STATES * NUM_BULLET_VECTORS)
				+ (sect * NUM_BULLET_VECTORS) + vect;

	cout << "thrust: " << (u32)aid.gene_accel[index][0] << endl;
	cout << "vector: " << (u32)aid.gene_heading[index][0] << endl;
}

int main(int argc, char** argv){
	po::options_description desc("Gene query options");
	desc.add_options()
		("help", "Show this help message")
		("generation", po::value<u32>(), "The generation number")
		("id", po::value<u32>(), "The id of the gene to view")
		("db", po::value<std::string>(), "The gene database, default is genes.kch")
		("dist", po::value<u32>(), "The distance state to check")
		("sect", po::value<u32>(), "The sector state to check")
		("vect", po::value<u32>(), "The vector state to check");

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if(vm.count("help")){
		cout << desc << endl;
		return 0;
	}

	// read in the parameters
	string db_path = "genes_cpu.kch";
	u32 id = 0;
	u32 generation = 0;
	u32 dist = 1;
	u32 sect = 0;
	u32 vect = 0;

	if(vm.count("dist")){
		dist = vm["dist"].as<u32>();
	}

	if(vm.count("sect")){
		sect = vm["sect"].as<u32>();
	}

	if(vm.count("vect")){
		vect = vm["vect"].as<u32>();
	}

	if(vm.count("id")){
		id = vm["id"].as<u32>();
	}

	if(vm.count("generation")){
		generation = vm["generation"].as<u32>();
	}

	if(vm.count("db")){
		db_path = vm["db"].as<string>();
	}

	perform_query(db_path, id, generation, dist, sect, vect);
	
	return 0;
}