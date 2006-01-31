#ifndef Tracker_SiInduceChargeOnStrips_H
#define Tracker_SiInduceChargeOnStrips_H

#include "SimTracker/SiStripDigitizer/interface/SiChargeCollectionDrifter.h"
#include "Geometry/TrackerSimAlgo/interface/StripGeomDetUnit.h"

#include<map>

class StripDet;
/**
 * Base class for the induction of signal on strips.
 */
class SiInduceChargeOnStrips{
 public:
  
  typedef map< int, float, less<int> > hit_map_type;
  
  
  virtual ~SiInduceChargeOnStrips() { }
  virtual hit_map_type induce(SiChargeCollectionDrifter::collection_type, const StripGeomDetUnit&) = 0 ;
};
#endif
