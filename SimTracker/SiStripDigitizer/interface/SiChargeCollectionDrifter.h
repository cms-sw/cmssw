#ifndef Tracker_SiChargeCollectionDrifter_H
#define Tracker_SiChargeCollectionDrifter_H

#include "Geometry/Vector/interface/LocalVector.h"
#include "SimTracker/SiStripDigitizer/interface/SignalPoint.h"
#include "Geometry/Vector/interface/LocalVector.h"
#include "SimTracker/SiStripDigitizer/interface/EnergyDepositUnit.h"

using namespace std;
#include<vector>
/**
 * Base class for the drifting of charges in the silicon.
 */
class SiChargeCollectionDrifter{
 public:  
  typedef vector <SignalPoint> collection_type;
  typedef vector <EnergyDepositUnit> ionization_type;

  virtual collection_type drift (const ionization_type, const LocalVector&) = 0;
};

#endif

