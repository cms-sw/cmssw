//
// Joanna Weng 08.2005
// Physics process for Gflash parameterisation
// modified by Soon Yung Jun, Dongwook Jang
// V.Ivanchenko rename the class, cleanup, and move
//              to SimG4Core/Application - 2012/08/14

#include "SimG4Core/Application/interface/ParametrisedEMPhysics.h"
#include "SimG4Core/Application/interface/GFlashEMShowerModel.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4FastSimulationManagerProcess.hh"
#include "G4ProcessManager.hh"

#include "G4LeptonConstructor.hh"
#include "G4MesonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4ShortLivedConstructor.hh"
#include "G4IonConstructor.hh"
#include "G4RegionStore.hh"

#include "G4EmProcessOptions.hh"

ParametrisedEMPhysics::ParametrisedEMPhysics(std::string name, const edm::ParameterSet & p) : G4VPhysicsConstructor(name), theParSet(p) 
{
  theEMShowerModel = 0;
  theHadShowerModel = 0;
}

ParametrisedEMPhysics::~ParametrisedEMPhysics() {
  delete theEMShowerModel;
  delete theHadShowerModel;
}

void ParametrisedEMPhysics::ConstructParticle() 
{
  G4LeptonConstructor pLeptonConstructor;
  pLeptonConstructor.ConstructParticle();

  G4MesonConstructor pMesonConstructor;
  pMesonConstructor.ConstructParticle();

  G4BaryonConstructor pBaryonConstructor;
  pBaryonConstructor.ConstructParticle();

  G4ShortLivedConstructor pShortLivedConstructor;
  pShortLivedConstructor.ConstructParticle();  
    
  G4IonConstructor pConstructor;
  pConstructor.ConstructParticle();  
}

void ParametrisedEMPhysics::ConstructProcess() {

  // GFlash part 
  bool gem  = theParSet.getParameter<bool>("GflashEcal");
  bool ghad = theParSet.getParameter<bool>("GflashHcal");

  if(gem || ghad) {
    edm::LogInfo("SimG4CoreApplication") 
      << "ParametrisedEMPhysics: GFlash Construct: " 
      << gem << "  " << ghad;
    G4FastSimulationManagerProcess * theFastSimulationManagerProcess = 
      new G4FastSimulationManagerProcess();
    theParticleIterator->reset();
    while ((*theParticleIterator)()) {
      G4ParticleDefinition * particle = theParticleIterator->value();
      G4ProcessManager * pmanager = particle->GetProcessManager();
      G4String pname = particle->GetParticleName();
      if(pname == "e-" || pname == "e+") {
	pmanager->AddProcess(theFastSimulationManagerProcess, -1, -1, 1);
      }
    }

    if(gem) {
      G4Region* aRegion = G4RegionStore::GetInstance()->GetRegion("EcalRegion");

      if(!aRegion){
	edm::LogInfo("SimG4CoreApplication") 
	  << "ParametrisedEMPhysics::ConstructProcess: " 
	  << "EcalRegion is not defined, GFlash will not be enabled for ECAL!";
	
      } else {

	//Electromagnetic Shower Model for ECAL
	theEMShowerModel = 
	  new GFlashEMShowerModel("GflashEMShowerModel",aRegion,theParSet);
	//std::cout << "GFlash is defined for EcalRegion" << std::endl;
      }    
    }
    if(ghad) {
      G4Region* aRegion = G4RegionStore::GetInstance()->GetRegion("HcalRegion");
      if(!aRegion) {
	edm::LogInfo("SimG4CoreApplication") 
	  << "ParametrisedEMPhysics::ConstructProcess: " 
	  << "HcalRegion is not defined, GFlash will not be enabled for HCAL!";
	
      } else {

	//Electromagnetic Shower Model for HCAL
	theHadShowerModel = 
	  new GFlashEMShowerModel("GflashHadShowerModel",aRegion,theParSet);
	//std::cout << "GFlash is defined for HcalRegion" << std::endl;
      }
    }
  }
  // Russian Roulette part 
  G4EmProcessOptions opt;
  double gamEcal = theParSet.getParameter<double>("RusRoEcalGamma");
  if(gamEcal < 1.0) {
    double gamEcalLim = theParSet.getParameter<double>("RusRoEcalGammaLimit")
      *CLHEP::MeV;
    if(gamEcalLim > 0.0) {
      opt.ActivateSecondaryBiasing("eBrem","EcalRegion",gamEcal,gamEcalLim);
      edm::LogInfo("SimG4CoreApplication") 
	<< "ParametrisedEMPhysics: Russian Roulette "
	<< " for gamma in ECAL Prob= " 
	<< gamEcal << "  Elimit(MeV)= " << gamEcalLim/CLHEP::MeV;
    }
  }
  double gamHcal = theParSet.getParameter<double>("RusRoHcalGamma");
  if(gamHcal < 1.0) {
    double gamHcalLim = theParSet.getParameter<double>("RusRoHcalGammaLimit")
      *CLHEP::MeV;
    if(gamHcalLim > 0.0) {
      opt.ActivateSecondaryBiasing("eBrem","HcalRegion",gamHcal,gamHcalLim);
      edm::LogInfo("SimG4CoreApplication") 
	<< "ParametrisedEMPhysics: Russian Roulette "
	<< " for gamma in HCAL Prob= " 
	<< gamHcal << "  Elimit(MeV)= " << gamHcalLim/CLHEP::MeV;
    }
  }
  double eEcal = theParSet.getParameter<double>("RusRoEcalElectron");
  if(eEcal < 1.0) {
    double eEcalLim = theParSet.getParameter<double>("RusRoEcalElectronLimit")
      *CLHEP::MeV;
    if(eEcalLim > 0.0) {
      opt.ActivateSecondaryBiasing("eIoni","EcalRegion",eEcal,eEcalLim);
      opt.ActivateSecondaryBiasing("hIoni","EcalRegion",eEcal,eEcalLim);
      opt.ActivateSecondaryBiasing("muIoni","EcalRegion",eEcal,eEcalLim);
      opt.ActivateSecondaryBiasing("ionIoni","EcalRegion",eEcal,eEcalLim);
      // opt.ActivateSecondaryBiasingForGamma("phot","EcalRegion",eEcal,eEcalLim);
      // opt.ActivateSecondaryBiasingForGamma("compt","EcalRegion",eEcal,eEcalLim);
      // opt.ActivateSecondaryBiasingForGamma("conv","EcalRegion",eEcal,eEcalLim);
      edm::LogInfo("SimG4CoreApplication") 
	<< "ParametrisedEMPhysics: Russian Roulette "
	<< " for electrons in ECAL Prob= " 
	<< eEcal << "  Elimit(MeV)= " << eEcalLim/CLHEP::MeV;
    }
  }
  double eHcal = theParSet.getParameter<double>("RusRoHcalElectron");
  if(eHcal < 1.0) {
    double eHcalLim = theParSet.getParameter<double>("RusRoHcalElectronLimit")
      *CLHEP::MeV;
    if(eHcalLim > 0.0) {
      opt.ActivateSecondaryBiasing("eIoni","HcalRegion",eHcal,eHcalLim);
      opt.ActivateSecondaryBiasing("hIoni","HcalRegion",eHcal,eHcalLim);
      opt.ActivateSecondaryBiasing("muIoni","HcalRegion",eHcal,eHcalLim);
      opt.ActivateSecondaryBiasing("ionIoni","HcalRegion",eHcal,eHcalLim);
      //opt.ActivateSecondaryBiasingForGamma("phot","HcalRegion",eHcal,eHcalLim);
      // opt.ActivateSecondaryBiasingForGamma("compt","HcalRegion",eHcal,eHcalLim);
      // opt.ActivateSecondaryBiasingForGamma("conv","HcalRegion",eHcal,eHcalLim);
      edm::LogInfo("SimG4CoreApplication") 
	<< "ParametrisedEMPhysics: Russian Roulette "
	<< " for electrons in HCAL Prob= " 
	<< eHcal << "  Elimit(MeV)= " << eHcalLim/CLHEP::MeV;
    }
  }
}
