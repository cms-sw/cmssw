#ifndef SimG4CMS_CaloMap_H
#define SimG4CMS_CaloMap_H

#include "SimG4Core/Notification/interface/TrackWithHistory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <map>

using namespace std;

class CaloMap {

public:

  static CaloMap* instance();
  ~CaloMap();

  void clear(const int evtID);
  void setTrack(const int id, TrackWithHistory* tk);
  TrackWithHistory* getTrack(const int id);
  typedef map<int,TrackWithHistory*> MapType;

private:

  CaloMap();

  static CaloMap* instance_; // For singleton behaviour
  MapType tkMap;

};

#endif
