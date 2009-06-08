#ifndef GflashHadronShowerProfile_H
#define GflashHadronShowerProfile_H 

#include "SimG4Core/GFlash/interface/GflashNameSpace.h"
#include "SimG4Core/GFlash/interface/GflashHistogram.h"
#include "SimG4Core/GFlash/interface/GflashTrajectory.h"
#include "SimG4Core/GFlash/interface/GflashShowino.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandGamma.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandChiSquare.h"

#include "G4VFastSimulationModel.hh"
#include "G4TouchableHandle.hh"
#include "G4Navigator.hh"
#include "G4Step.hh"
#include "SimG4Core/Notification/interface/SimActivityRegistry.h"

#include <vector>

class GflashHadronShowerProfile 
{
public:
  //-------------------------
  // Constructor, destructor
  //-------------------------
  GflashHadronShowerProfile (edm::ParameterSet parSet);
  virtual ~GflashHadronShowerProfile ();

  virtual void loadParameters(const G4FastTrack& fastTrack);
  void hadronicParameterization(const G4FastTrack& fastTrack);
  G4int getShowerType() { return theShowerType; }

protected:
  void setShowerType(const G4FastTrack& fastTrack);
  G4double longitudinalProfile(G4double showerDepth, G4double pathLength);
  G4double hoProfile(G4double pathLength, G4double refDepth);
  void doCholeskyReduction(G4double **cc, G4double **vv, const G4int ndim);
  G4double* getFluctuationVector(G4double *lowTriangle);
  void setEnergyScale(G4double einc, G4ThreeVector ssp);

  G4int getNumberOfSpots(Gflash::CalorimeterNumber kCalor);
  G4double medianLateralArm(G4double depth, Gflash::CalorimeterNumber kCalor);
  G4ThreeVector locateSpotPosition(GflashTrajectoryPoint& point, G4double lateralArm);
  void updateGflashStep(G4ThreeVector position, G4double time);

  G4double fTanh(G4double einc, const G4double *par);
  G4double fLnE1(G4double einc, const G4double *par);
  G4double depthScale(G4double ssp, G4double ssp0, G4double length);
  G4double gammaProfile(G4double alpha, G4double beta, G4double depth, G4double lengthUnit);
  G4double twoGammaProfile(G4double *par, G4double depth, Gflash::CalorimeterNumber kIndex);

  SimActivityRegistry::G4StepSignal gflash_g4StepSignal;

protected:
  edm::ParameterSet theParSet;
  G4double theBField;
  G4bool theGflashHcalOuter;

  G4int theShowerType ; 

  G4Step *theGflashStep; 
  GflashShowino *theShowino; 
  G4Navigator *theGflashNavigator;
  G4TouchableHandle  theGflashTouchableHandle;

  GflashHistogram* theHisto;

  CLHEP::RandGaussQ* theRandGauss;
  CLHEP::RandGamma*  theRandGamma;
  CLHEP::RandPoissonQ*  theRandPoissonQ;
  CLHEP::RandChiSquare* theRandChiSquare;

  G4double energyScale[Gflash::kNumberCalorimeter]; 
  G4double averageSpotEnergy[Gflash::kNumberCalorimeter]; 
  G4double longEcal[Gflash::NPar];  
  G4double longHcal[Gflash::NPar];  
  G4double lateralPar[Gflash::kNumberCalorimeter][Gflash::Nrpar]; 

};

#endif



