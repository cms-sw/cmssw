#ifndef Tracker_SiInduceChargeOnStrips_H
#define Tracker_SiInduceChargeOnStrips_H

#include "SimTracker/SiStripDigitizer/interface/SiChargeCollectionDrifter.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "SimTracker/SiStripDigitizer/interface/SiPileUpSignals.h"

#include<map>

class StripDet;
/**
 * Base class for the induction of signal on strips.
 */
class SiInduceChargeOnStrips{
 public:
  
  typedef std::map< int, float, std::less<int> > hit_map_type;
  
  
  virtual ~SiInduceChargeOnStrips() { }
  virtual void induce(SiChargeCollectionDrifter::collection_type, const StripGeomDetUnit&, 
		      SiPileUpSignals::signal_map_type &, SiPileUpSignals::signal_map_type &) = 0 ;
};
#endif
