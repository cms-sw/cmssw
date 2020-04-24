#ifndef Tracker_SiInduceChargeOnStrips_H
#define Tracker_SiInduceChargeOnStrips_H

#include "SiChargeCollectionDrifter.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "SimTracker/SiStripDigitizer/interface/SiPileUpSignals.h"

#include<map>

class TrackerTopology;
class StripDet;
/**
 * Base class for the induction of signal on strips.
 * Given a SignalPoint, computes the charge on each strip.
 */


class SiInduceChargeOnStrips{
public:
  
  virtual ~SiInduceChargeOnStrips() { }
  virtual void induce(const SiChargeCollectionDrifter::collection_type&, const StripGeomDetUnit&, 
		      std::vector<float>&, size_t&, size_t&,
		      const TrackerTopology *tTopo) const=0;

};
#endif
