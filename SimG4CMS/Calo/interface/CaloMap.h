#ifndef SimG4CMS_CaloMap_H
#define SimG4CMS_CaloMap_H

#include "SimG4Core/Notification/interface/TrackWithHistory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <map>

class CaloMap {

public:

  static CaloMap* instance();
  ~CaloMap();

  void clear(const int evtID);
  void setTrack(const int id, TrackWithHistory* tk);
  TrackWithHistory* getTrack(const int id);
  typedef std::map<int,TrackWithHistory*> MapType;

private:

  CaloMap();

  static CaloMap* instance_; // For singleton behaviour
  MapType tkMap;

};

#endif
