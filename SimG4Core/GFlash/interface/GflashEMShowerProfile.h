#ifndef GflashEMShowerProfile_H
#define GflashEMShowerProfile_H 

#include "SimG4Core/GFlash/interface/GflashTrajectory.h"
#include "SimG4Core/GFlash/interface/GflashNameSpace.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
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
  GflashEMShowerProfile (G4Region* envelope, edm::ParameterSet parSet);
  ~GflashEMShowerProfile ();

  void clearSpotList() { aEnergySpotList.clear(); }
  void parameterization(const G4FastTrack& fastTrack);
  std::vector<GflashEnergySpot> &getEnergySpotList() {return aEnergySpotList;}; 
  Gflash::CalorimeterNumber getCalorimeterNumber(const G4ThreeVector position);

private:  

  edm::ParameterSet theParSet;
  std::vector<GflashEnergySpot> aEnergySpotList;

  GflashHistogram* theHisto;
  GflashTrajectory* theHelix;

  CLHEP::RandGaussQ* theRandGauss;
  Gflash::CalorimeterNumber jCalorimeter;
  G4double theBField;
  G4double theGflash5x5EnergyScale_a;
  G4double theGflash5x5EnergyScale_b;

  // temporary addition for tuning parameters
  std::vector<double> theEMLateral_pList;

};

#endif




