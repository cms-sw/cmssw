#ifndef GflashEMShowerProfile_H
#define GflashEMShowerProfile_H 

#include "G4VFastSimulationModel.hh"
#include "CLHEP/Random/RandGaussQ.h"
#include <vector>

class GflashEnergySpot;
class GflashHistogram;

class GflashEMShowerProfile 
{
public:
  //-------------------------
  // Constructor, destructor
  //-------------------------
  GflashEMShowerProfile (G4Region* envelope);
  ~GflashEMShowerProfile ();

  void clearSpotList() { aEnergySpotList.clear(); }
  void parameterization(const G4FastTrack& fastTrack);
  std::vector<GflashEnergySpot> &getEnergySpotList() {return aEnergySpotList;}; 

private:  

  std::vector<GflashEnergySpot> aEnergySpotList;

  GflashHistogram* theHisto;
  CLHEP::RandGaussQ* theRandGauss;
};

#endif




