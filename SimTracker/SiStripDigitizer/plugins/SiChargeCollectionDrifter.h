#ifndef Tracker_SiChargeCollectionDrifter_H
#define Tracker_SiChargeCollectionDrifter_H

#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "SignalPoint.h"
#include "EnergyDepositUnit.h"

#include<vector>
/**
 * Base class for the drifting of charges in the silicon.
 */
class SiChargeCollectionDrifter{
 public:  
  typedef std::vector <SignalPoint> collection_type;
  typedef std::vector <EnergyDepositUnit> ionization_type;

  virtual ~SiChargeCollectionDrifter() { }
  virtual collection_type drift (const ionization_type, const LocalVector&,double,double) = 0;
};

#endif

