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

  bool gem  = theParSet.getParameter<bool>("GflashEcal");
  bool ghad = theParSet.getParameter<bool>("GflashHcal");
  edm::LogInfo("SimG4CoreApplication") 
    << "ParametrisedEMPhysics::ConstructProcess: GFlash Construct: " 
    << gem << "  " << ghad;

  if(gem || ghad) {
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
}
