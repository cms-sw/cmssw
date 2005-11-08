#include "SimG4CMS/Calo/interface/CaloMap.h"

//UserVerbosity CaloMap::cout("CaloMap","silent","CaloSD");

// Initialise the singleton pointer
CaloMap* CaloMap::instance_ = 0;

CaloMap* CaloMap::instance() {
  if (instance_ == 0)
    instance_ = new CaloMap;
  return instance_;
}

CaloMap::~CaloMap() {
  if (verbosity > 0) std::cout << "Deleting CaloMap" << std::endl;
}

void CaloMap::clear(const int evtID) {

  tkMap.erase (tkMap.begin(), tkMap.end());
  if (verbosity > 1) 
    std::cout << "CaloMap: Erase TrackWithHistory map for event " << evtID
	      << std::endl;
}

void CaloMap::setTrack(const int id, TrackWithHistory* tk) {

  if (verbosity > 1) 
    std::cout << "CaloMap: ID " << id << " Current trkHistory " << tkMap[id]
	      << " New trkHistory " << tk << std::endl;
  tkMap[id] = tk;
}

TrackWithHistory* CaloMap::getTrack(const int id) {

  return tkMap[id];
}

void CaloMap::setVerbosity(const int iv) {verbosity = iv;}

CaloMap::CaloMap() : verbosity(0) {  

  std::cout << "CaloMap: Initialised" << std::endl;
}


