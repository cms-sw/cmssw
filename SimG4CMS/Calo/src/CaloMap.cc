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
  edm::LogInfo("CaloSimInfo") << "Deleting CaloMap";
}

void CaloMap::clear(const int evtID) {

  tkMap.erase (tkMap.begin(), tkMap.end());
  LogDebug("CaloSimInfo") << "CaloMap: Erase TrackWithHistory map for event " << evtID;
}

void CaloMap::setTrack(const int id, TrackWithHistory* tk) {

  LogDebug("CaloSimInfo") << "CaloMap: ID " << id << " Current trkHistory " << tkMap[id]
                          << " New trkHistory " << tk;
  tkMap[id] = tk;
}

TrackWithHistory* CaloMap::getTrack(const int id) {

  return tkMap[id];
}

void CaloMap::setVerbosity(const int iv) {verbosity = iv;}

CaloMap::CaloMap() : verbosity(0) {  

  edm::LogInfo("CaloSimInfo") << "CaloMap: Initialised";
}


