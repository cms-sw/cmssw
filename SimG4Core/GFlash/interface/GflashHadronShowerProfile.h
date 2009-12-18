#ifndef GflashHadronShowerProfile_H
#define GflashHadronShowerProfile_H 

#include "SimG4Core/GFlash/interface/GflashNameSpace.h"
#include "SimG4Core/GFlash/interface/GflashHistogram.h"
#include "SimG4Core/GFlash/interface/GflashTrajectory.h"
#include "SimG4Core/GFlash/interface/GflashShowino.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4VFastSimulationModel.hh"
#include "G4TouchableHandle.hh"
#include "G4Navigator.hh"
#include "G4Step.hh"
#include "SimG4Core/Notification/interface/SimActivityRegistry.h"

#include <vector>

class FSimTrack;
class EcalHitMaker;
class HcalHitMaker;

class GflashHadronShowerProfile 
{
public:
  //-------------------------
  // Constructor, destructor
  //-------------------------
  GflashHadronShowerProfile (edm::ParameterSet parSet);
  virtual ~GflashHadronShowerProfile ();

  void initialize(const G4FastTrack& fastTrack);
  virtual void loadParameters();
  void hadronicParameterization();
  GflashShowino* getGflashShowino() { return theShowino; }
  void initFastSimCaloHit(EcalHitMaker *aEcalHitMaker,
                          HcalHitMaker *aHcalHitMaker);

protected:
  G4double longitudinalProfile(G4double showerDepth, G4double pathLength);
  G4double hoProfile(G4double pathLength, G4double refDepth);
  void doCholeskyReduction(G4double **cc, G4double **vv, const G4int ndim);
  void getFluctuationVector(G4double *lowTriangle, G4double *correlationVector);
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
  G4bool theExportToFastSim;

  G4Step *theGflashStep; 
  GflashShowino *theShowino; 
  G4Navigator *theGflashNavigator;
  G4TouchableHandle  theGflashTouchableHandle;

  GflashHistogram* theHisto;

  G4double energyScale[Gflash::kNumberCalorimeter]; 
  G4double averageSpotEnergy[Gflash::kNumberCalorimeter]; 
  G4double longEcal[Gflash::NPar];  
  G4double longHcal[Gflash::NPar];  
  G4double lateralPar[Gflash::kNumberCalorimeter][Gflash::Nrpar]; 

  //FastSim related Output
  EcalHitMaker* theEcalHitMaker;
  HcalHitMaker* theHcalHitMaker;

};

#endif



