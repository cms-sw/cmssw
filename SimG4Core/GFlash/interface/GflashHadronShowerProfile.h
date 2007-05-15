#ifndef GflashHadronShowerProfile_H
#define GflashHadronShowerProfile_H 

#include "SimG4Core/GFlash/interface/GflashMediaMap.h"
#include "SimG4Core/GFlash/interface/GflashCalorimeterNumber.h"

#include "GFlashEnergySpot.hh"
#include "G4VFastSimulationModel.hh"
#include "G4Step.hh"
#include "G4TouchableHandle.hh"
#include <vector>

class GflashHadronShowerProfile 
{
public:
  //-------------------------
  // Constructor, destructor
  //-------------------------
  GflashHadronShowerProfile (G4Region* envelope);
  ~GflashHadronShowerProfile ();

  GflashCalorimeterNumber getCalorimeterNumber(const G4FastTrack& fastTrack);
  void hadronicParameterization(const G4FastTrack& fastTrack);
  std::vector<GFlashEnergySpot> getEnergySpotList() {return aEnergySpotList;}; 

private:
  void loadParameters(G4double einc);
  void fillFluctuationVector(G4double *RMX);
  void doCholeskyReduction(G4double **cc, G4double **vv, const G4int ndim);
  void samplingFluctuation(G4double &de, G4double einc);
  inline GflashCalorimeterNumber getCalorimeterNumber() {return jCalorimeter;}

private:  

  G4int showerType ; 
  GflashCalorimeterNumber jCalorimeter ;
  std::vector<GFlashEnergySpot> aEnergySpotList;

  G4double energyToDeposit; 
  //lateral and longitudinal parameters
  G4double lateralPar[4]; 
  G4double alpha[4];  
  G4double beta[4];
  G4double frac[4];

  GflashMediaMap* theMediaMap;

};

#endif




