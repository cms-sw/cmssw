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

ParametrisedEMPhysics::ParametrisedEMPhysics(std::string name, const edm::ParameterSet & p) 
  : G4VPhysicsConstructor(name), theParSet(p) 
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
  const G4String rname[8] = {"EcalRegion", "HcalRegion", "QuadRegion", "MuonIron",
			     "PreshowerRegion","CastorRegion","BeamPipeOutsideRegion",
			     "DefaultRegionForTheWorld"};
  G4double rrfact[8] = { 1.0 };

  // Russian roulette for gamma
  G4double energyLim = theParSet.getParameter<double>("RusRoGammaEnergyLimit")*MeV;
  if(energyLim > 0.0) {
    rrfact[0] = theParSet.getParameter<double>("RusRoEcalGamma");
    rrfact[1] = theParSet.getParameter<double>("RusRoHcalGamma");
    rrfact[2] = theParSet.getParameter<double>("RusRoQuadGamma");
    rrfact[3] = theParSet.getParameter<double>("RusRoMuonIronGamma");
    rrfact[4] = theParSet.getParameter<double>("RusRoPreShowerGamma");
    rrfact[5] = theParSet.getParameter<double>("RusRoCastorGamma");
    rrfact[6] = theParSet.getParameter<double>("RusRoBeamPipeOutGamma");
    rrfact[7] = theParSet.getParameter<double>("RusRoWorldGamma");
    for(int i=0; i<8; ++i) {
      if(rrfact[i] < 1.0) {
	opt.ActivateSecondaryBiasing("eBrem",rname[i],rrfact[i],energyLim);
	edm::LogInfo("SimG4CoreApplication") 
	  << "ParametrisedEMPhysics: Russian Roulette"
	  << " for gamma Prob= " << rrfact[i]  
	  << " Elimit(MeV)= " << energyLim/CLHEP::MeV
	  << " inside " << rname[i];
      }
    }
  }
  // Russian roulette for e-
  energyLim = theParSet.getParameter<double>("RusRoElectronEnergyLimit")*MeV;
  if(energyLim > 0.0) {
    rrfact[0] = theParSet.getParameter<double>("RusRoEcalElectron");
    rrfact[1] = theParSet.getParameter<double>("RusRoHcalElectron");
    rrfact[2] = theParSet.getParameter<double>("RusRoQuadElectron");
    rrfact[3] = theParSet.getParameter<double>("RusRoMuonIronElectron");
    rrfact[4] = theParSet.getParameter<double>("RusRoPreShowerElectron");
    rrfact[5] = theParSet.getParameter<double>("RusRoCastorElectron");
    rrfact[6] = theParSet.getParameter<double>("RusRoBeamPipeOutElectron");
    rrfact[7] = theParSet.getParameter<double>("RusRoWorldElectron");
    for(int i=0; i<8; ++i) {
      if(rrfact[i] < 1.0) {
	opt.ActivateSecondaryBiasing("eIoni",rname[i],rrfact[i],energyLim);
	opt.ActivateSecondaryBiasing("hIoni",rname[i],rrfact[i],energyLim);
	opt.ActivateSecondaryBiasing("muIoni",rname[i],rrfact[i],energyLim);
	opt.ActivateSecondaryBiasing("ionIoni",rname[i],rrfact[i],energyLim);
	opt.ActivateSecondaryBiasingForGamma("phot",rname[i],rrfact[i],energyLim);
	opt.ActivateSecondaryBiasingForGamma("compt",rname[i],rrfact[i],energyLim);
	opt.ActivateSecondaryBiasingForGamma("conv",rname[i],rrfact[i],energyLim);
	edm::LogInfo("SimG4CoreApplication") 
	  << "ParametrisedEMPhysics: Russian Roulette"
	  << " for e- Prob= " << rrfact[i]  
	  << " Elimit(MeV)= " << energyLim/CLHEP::MeV
	  << " inside " << rname[i];
      }
    }
  }
}
