#ifndef Tracker_SiChargeDivider_H
#define Tracker_SiChargeDivider_H

#include "SimTracker/SiStripDigitizer/interface/EnergyDepositUnit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

using namespace std;
#include <vector>
/**
 * Base class for the division of a Geant energy deposit in smaller elementary charges inside the silicon
 */
class SiChargeDivider{
 public:
  
  typedef vector< EnergyDepositUnit > ionization_type;
  
  virtual ~SiChargeDivider() { }
  virtual ionization_type divide(const PSimHit&, const StripGeomDetUnit& det) = 0;
};


#endif
