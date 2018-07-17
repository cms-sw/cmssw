#ifndef SimG4CMS_HFGflash_h
#define SimG4CMS_HFGflash_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimGeneral/GFlash/interface/GflashEMShowerProfile.h"
#include "SimGeneral/GFlash/interface/GflashTrajectory.h"
#include "SimGeneral/GFlash/interface/GflashNameSpace.h"

#include "G4VFastSimulationModel.hh"
#include "G4ParticleTable.hh"
#include "G4ThreeVector.hh"
#include "G4FastTrack.hh"
#include "G4TouchableHandle.hh"
#include "G4Navigator.hh"

#include <string>
#include <vector>

#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TProfile.h>

class G4Step;

class HFGflash {

public:    

  HFGflash(edm::ParameterSet const & p);
  virtual ~HFGflash();

  struct Hit {
    Hit() {}
    G4ThreeVector       position;
    int                 depth = 0;
    double              time = 0.;
    double              edep = 0.;
    double              pez = 0.;
  };

  std::vector<Hit> gfParameterization(const G4Step * aStep, bool onlyLong=false);

private:    

  GflashTrajectory* theHelix;
  G4Step *theGflashStep;
  G4Navigator *theGflashNavigator;
  G4TouchableHandle  theGflashTouchableHandle;

  Gflash::CalorimeterNumber jCalorimeter;

  bool     theWatcherOn;
  bool     theFillHisto;
  G4double theBField;

  G4int    showerType ;
  G4double energyToDeposit; 
  G4double energyScale[Gflash::kNumberCalorimeter]; 
  G4double longHcal[Gflash::NPar];  
  G4double longEcal[Gflash::NPar];  
  G4double lateralPar[4]; 

  TH1F     *em_incE, *em_ssp_rho, *em_ssp_z, *em_long, *em_lateral;
  TH1F     *em_long_sd, *em_lateral_sd, *em_nSpots_sd, *em_ze_ratio;
  TH2F     *em_2d, *em_2d_sd, *em_ze, *em_ratio, *em_ratio_selected;
};

#endif // HFGflash_h
