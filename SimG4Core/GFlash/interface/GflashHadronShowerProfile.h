#ifndef GflashHadronShowerProfile_H
#define GflashHadronShowerProfile_H 

//#include "SimG4Core/GFlash/interface/GflashMediaMap.h"
//#include "SimG4Core/GFlash/interface/GflashCalorimeterNumber.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandGamma.h"

#include "G4VFastSimulationModel.hh"
#include "G4Step.hh"
#include "G4TouchableHandle.hh"

#include "SimG4Core/GFlash/interface/GflashNameSpace.h"

#include <vector>

class GflashEnergySpot;
class GflashHistogram;

class GflashHadronShowerProfile 
{
public:
  //-------------------------
  // Constructor, destructor
  //-------------------------
  GflashHadronShowerProfile (G4Region* envelope);
  ~GflashHadronShowerProfile ();

  Gflash::CalorimeterNumber getCalorimeterNumber(const G4FastTrack& fastTrack);
  void hadronicParameterization(const G4FastTrack& fastTrack);
  std::vector<GflashEnergySpot>& getEnergySpotList() {return aEnergySpotList;}; 

private:
  void loadParameters(G4double einc);
  void fillFluctuationVector();
  void doCholeskyReduction(G4double **cc, G4double **vv, const G4int ndim);
  void samplingFluctuation(G4double &de, G4double einc);
  inline Gflash:: CalorimeterNumber getCalorimeterNumber() {return jCalorimeter;}

  G4bool insideSampling(const G4ThreeVector pos);

private:  

  //correlation vector: size = 190+8*9/2+8
  G4double RMX[234];

  G4int showerType ; 
  Gflash::CalorimeterNumber jCalorimeter ;
  std::vector<GflashEnergySpot> aEnergySpotList;

  G4double energyToDeposit; 
  //lateral and longitudinal parameters
  G4double lateralPar[4]; 
  G4double alpha[4];  
  G4double beta[4];
  G4double frac[4];

  //  G4Step* theGflashStep;

  //  GflashMediaMap* theMediaMap;
  GflashHistogram* theHisto;

  CLHEP::RandGaussQ* theRandGauss;
  CLHEP::RandGamma*  theRandGamma;
};

#endif




