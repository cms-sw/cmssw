#ifndef Tracker_SiChargeCollectionDrifter_H
#define Tracker_SiChargeCollectionDrifter_H

#include "Geometry/Vector/interface/LocalVector.h"
#include "SimTracker/SiStripDigitizer/interface/SignalPoint.h"
#include "Geometry/Vector/interface/LocalVector.h"
#include "SimTracker/SiStripDigitizer/interface/EnergyDepositUnit.h"

#include<vector>
/**
 * Base class for the drifting of charges in the silicon.
 */
class SiChargeCollectionDrifter{
 public:  
  typedef std::vector <SignalPoint> collection_type;
  typedef std::vector <EnergyDepositUnit> ionization_type;

  virtual ~SiChargeCollectionDrifter() { }
  virtual collection_type drift (const ionization_type, const LocalVector&) = 0;
};

#endif

