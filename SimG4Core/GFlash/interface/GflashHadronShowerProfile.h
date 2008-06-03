#ifndef GflashHadronShowerProfile_H
#define GflashHadronShowerProfile_H 

#include "SimG4Core/GFlash/interface/GflashNameSpace.h"
#include "SimG4Core/GFlash/interface/GflashTrajectory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandGamma.h"

#include "G4VFastSimulationModel.hh"
#include "G4Step.hh"
#include "G4TouchableHandle.hh"

#include <vector>

class GflashEnergySpot;
class GflashHistogram;

class GflashHadronShowerProfile 
{
public:
  //-------------------------
  // Constructor, destructor
  //-------------------------
  GflashHadronShowerProfile (G4Region* envelope, edm::ParameterSet parSet);
  ~GflashHadronShowerProfile ();

  Gflash::CalorimeterNumber getCalorimeterNumber(const G4ThreeVector position);
  void clearSpotList() { aEnergySpotList.clear(); }
  void hadronicParameterization(const G4FastTrack& fastTrack);
  std::vector<GflashEnergySpot>& getEnergySpotList() {return aEnergySpotList;}; 

private:
  void loadParameters(const G4FastTrack& fastTrack);
  G4double longitudinalProfile(G4double showerDepth, G4double pathLength, G4double transDepth);
  void samplingFluctuation(G4double &de, G4double einc);
  G4bool insideSampling(const G4ThreeVector pos);
  void doCholeskyReduction(G4double **cc, G4double **vv, const G4int ndim);
  void fillFluctuationVector();

private:  

  edm::ParameterSet theParSet;
  G4int showerType ; 
  Gflash::CalorimeterNumber jCalorimeter ;
  std::vector<GflashEnergySpot> aEnergySpotList;
  G4double theBField;

  G4double energyToDeposit; 
  //lateral and longitudinal parameters
  G4double correlationVector[Gflash::NRegion*Gflash::NxN*(Gflash::NxN+1)/2]; //21*3 = 63
  G4double longPar[Gflash::NRegion][Gflash::NxN];  
  //  G4double longPar2[6];  
  //  G4double longPar3[6];  
  G4double lateralPar[4]; 

  //  G4double longSigma1[6];  
  G4double longSigma[Gflash::NRegion][Gflash::NxN];  

  //  GflashMediaMap* theMediaMap;
  GflashHistogram* theHisto;
  GflashTrajectory* theHelix;

  CLHEP::RandGaussQ* theRandGauss;
  CLHEP::RandGamma*  theRandGamma;
};

#endif




