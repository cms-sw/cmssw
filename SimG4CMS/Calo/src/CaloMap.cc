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
  cout << "Deleting CaloMap" << endl;
}

void CaloMap::clear(const int evtID) {

  tkMap.erase (tkMap.begin(), tkMap.end());
  cout << "CaloMap: Erase TrackWithHistory map for event " << evtID
	       << endl;
}

void CaloMap::setTrack(const int id, TrackWithHistory* tk) {

  cout << "CaloMap: ID " << id << " Current trkHistory " << tkMap[id]
	       << " New trkHistory " << tk << endl;
  tkMap[id] = tk;
}

TrackWithHistory* CaloMap::getTrack(const int id) {

  return tkMap[id];
}

CaloMap::CaloMap() {  

  cout << "CaloMap: Initialised" << endl;
}


