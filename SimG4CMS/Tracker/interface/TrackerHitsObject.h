#ifndef TrackerHitsObject_H
#define TrackerHitsObject_H

#include "SimDataFormats/SimHitMaker/interface/TrackingSlaveSD.h"

class TrackerHitsObject {
public:
  TrackerHitsObject(std::string, TrackingSlaveSD::Collection&);
  std::string name() { return _name; }
  TrackingSlaveSD::Collection& hits() { return _hits; }

private:
  TrackingSlaveSD::Collection& _hits;
  std::string _name;
};

#endif
