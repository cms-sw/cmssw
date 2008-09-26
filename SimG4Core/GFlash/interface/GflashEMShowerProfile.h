#ifndef GflashEMShowerProfile_H
#define GflashEMShowerProfile_H 

#include "SimG4Core/GFlash/interface/GflashTrajectory.h"
#include "SimG4Core/GFlash/interface/GflashNameSpace.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "G4VFastSimulationModel.hh"
#include "G4TouchableHandle.hh"
#include "G4Navigator.hh"
#include "CLHEP/Random/RandGaussQ.h"
#include <vector>

class GflashHistogram;
class G4Step;

class GflashEMShowerProfile 
{
public:
  //-------------------------
  // Constructor, destructor
  //-------------------------
  GflashEMShowerProfile (G4Region* envelope, edm::ParameterSet parSet);
  ~GflashEMShowerProfile ();

  void parameterization(const G4FastTrack& fastTrack);
  Gflash::CalorimeterNumber getCalorimeterNumber(const G4ThreeVector position);

private:  

  edm::ParameterSet theParSet;

  GflashHistogram* theHisto;
  GflashTrajectory* theHelix;
  G4Step *theGflashStep;
  G4Navigator *theGflashNavigator;
  G4TouchableHandle  theGflashTouchableHandle;

  CLHEP::RandGaussQ* theRandGauss;
  Gflash::CalorimeterNumber jCalorimeter;
  G4double theBField;
  G4double theGflash5x5EnergyScale_a;
  G4double theGflash5x5EnergyScale_b;

  // temporary addition for tuning parameters
  std::vector<double> theEMLateral_pList;

};

#endif




