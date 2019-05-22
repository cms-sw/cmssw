#include "SimG4CMS/Tracker/interface/TrackerHitsObject.h"

TrackerHitsObject::TrackerHitsObject(std::string n, TrackingSlaveSD::Collection& h) : _hits(h), _name(n) {}
