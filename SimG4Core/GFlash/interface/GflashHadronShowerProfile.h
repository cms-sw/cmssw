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
#include "G4Navigator.hh"

#include "SimG4Core/Notification/interface/SimActivityRegistry.h"


#include <vector>

class GflashHistogram;
class G4Step;

class GflashHadronShowerProfile 
{
public:
  //-------------------------
  // Constructor, destructor
  //-------------------------
  GflashHadronShowerProfile (G4Region* envelope, edm::ParameterSet parSet);
  ~GflashHadronShowerProfile ();

  void hadronicParameterization(const G4FastTrack& fastTrack);

private:
  void loadParameters(const G4FastTrack& fastTrack);
  G4double longitudinalProfile(G4double showerDepth, G4double pathLength, G4double transDepth, 
			       const G4ThreeVector pos,G4double einc);
  void samplingFluctuation(G4double &de, G4double einc, Gflash::CalorimeterNumber whichCalor);
  G4bool insideSampling(Gflash::CalorimeterNumber whichCalor);
  void doCholeskyReduction(G4double **cc, G4double **vv, const G4int ndim);
  G4double* getFluctuationVector(G4double *lowTriangle);
  void setEnergyScale(G4double einc, G4ThreeVector ssp);
  G4int getNumberOfSpots(G4double einc, Gflash::CalorimeterNumber whichCalor);
  G4double fTanh(G4double einc, const G4double *par);
  G4double fLnE1(G4double einc, const G4double *par);
  G4double depthScale(G4double ssp, G4double ssp0, G4double length);
  G4double twoGammaProfile(G4double *par, G4double depth, Gflash::CalorimeterNumber kIndex);
  G4double gammaProfile(G4double alpha, G4double beta, G4double depth, G4double lengthUnit);

  SimActivityRegistry::G4StepSignal gflash_g4StepSignal;


private:  

  edm::ParameterSet theParSet;
  G4int showerType ; 
  Gflash::CalorimeterNumber jCalorimeter ;
  G4double theBField;

  G4double energyToDeposit; 
  G4double energyScale[Gflash::kNumberCalorimeter]; 
  G4double averageSpotEnergy[Gflash::kNumberCalorimeter]; 
  //lateral and longitudinal parameters
  //  G4double correlationVector[Gflash::NRegion*Gflash::NPar*(Gflash::NPar+1)/2]; //21*3 = 63
  //  G4double longPar[Gflash::NRegion][Gflash::NPar];  
  G4double longEcal[Gflash::NPar];  
  G4double longHcal[Gflash::NPar];  
  //  G4double longPar2[6];  
  //  G4double longPar3[6];  
  G4double lateralPar[Gflash::kNumberCalorimeter][Gflash::Nrpar]; 

  //  G4double longSigma1[6];  
  //  G4double longSigma[Gflash::NRegion][Gflash::NPar];  

  //  GflashMediaMap* theMediaMap;
  GflashHistogram* theHisto;
  GflashTrajectory* theHelix;
  G4Step *theGflashStep; 
  G4Navigator *theGflashNavigator;
  G4TouchableHandle  theGflashTouchableHandle;

  CLHEP::RandGaussQ* theRandGauss;
  CLHEP::RandGamma*  theRandGamma;
};

#endif




