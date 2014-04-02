//
// Joanna Weng 08.2005
// Physics process for Gflash parameterisation
// modified by Soon Yung Jun, Dongwook Jang
// V.Ivanchenko rename the class, cleanup, and move
//              to SimG4Core/Application - 2012/08/14

#include "SimG4Core/Application/interface/ParametrisedEMPhysics.h"
#include "SimG4Core/Application/interface/GFlashEMShowerModel.h"
#include "SimG4Core/Application/interface/ElectronLimiter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4FastSimulationManagerProcess.hh"
#include "G4ProcessManager.hh"

#include "G4LeptonConstructor.hh"
#include "G4MesonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4ShortLivedConstructor.hh"
#include "G4IonConstructor.hh"
#include "G4RegionStore.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"

#include "G4EmProcessOptions.hh"
#include "G4PhysicsListHelper.hh"
#include "G4SystemOfUnits.hh"

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
    aParticleIterator->reset();
    while ((*aParticleIterator)()) {
      G4ParticleDefinition * particle = aParticleIterator->value();
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
  const G4int NREG = 6; 
  const G4String rname[NREG] = {"EcalRegion", "HcalRegion", "MuonIron",
				"PreshowerRegion","CastorRegion",
				"DefaultRegionForTheWorld"};
  G4double rrfact[NREG] = { 1.0 };

  // Russian roulette for e-
  double energyLim = theParSet.getParameter<double>("RusRoElectronEnergyLimit")*MeV;
  if(energyLim > 0.0) {
    rrfact[0] = theParSet.getParameter<double>("RusRoEcalElectron");
    rrfact[1] = theParSet.getParameter<double>("RusRoHcalElectron");
    rrfact[2] = theParSet.getParameter<double>("RusRoMuonIronElectron");
    rrfact[3] = theParSet.getParameter<double>("RusRoPreShowerElectron");
    rrfact[4] = theParSet.getParameter<double>("RusRoCastorElectron");
    rrfact[5] = theParSet.getParameter<double>("RusRoWorldElectron");
    for(int i=0; i<NREG; ++i) {
      if(rrfact[i] < 1.0) {
	opt.ActivateSecondaryBiasing("eIoni",rname[i],rrfact[i],energyLim);
	opt.ActivateSecondaryBiasing("hIoni",rname[i],rrfact[i],energyLim);
	//opt.ActivateSecondaryBiasing("muIoni",rname[i],rrfact[i],energyLim);
	//opt.ActivateSecondaryBiasing("ionIoni",rname[i],rrfact[i],energyLim);
	//opt.ActivateSecondaryBiasingForGamma("phot",rname[i],rrfact[i],energyLim);
	//opt.ActivateSecondaryBiasingForGamma("compt",rname[i],rrfact[i],energyLim);
	//opt.ActivateSecondaryBiasingForGamma("conv",rname[i],rrfact[i],energyLim);
	edm::LogInfo("SimG4CoreApplication") 
	  << "ParametrisedEMPhysics: Russian Roulette"
	  << " for e- Prob= " << rrfact[i]  
	  << " Elimit(MeV)= " << energyLim/CLHEP::MeV
	  << " inside " << rname[i];
      }
    }
  }

  // Step limiters for e+-
  bool eLimiter = theParSet.getParameter<bool>("ElectronStepLimit");
  bool rLimiter = theParSet.getParameter<bool>("ElectronRangeTest");
  bool pLimiter = theParSet.getParameter<bool>("PositronStepLimit");

  if(eLimiter || rLimiter ||  pLimiter) {
    G4PhysicsListHelper* ph = G4PhysicsListHelper::GetPhysicsListHelper();

    if(eLimiter || rLimiter) {
      theElectronLimiter = new ElectronLimiter(theParSet);
      theElectronLimiter->SetRangeCheckFlag(rLimiter);
      theElectronLimiter->SetFieldCheckFlag(eLimiter);
      ph->RegisterProcess(theElectronLimiter, G4Electron::Electron());
    }
  
    if(pLimiter){
      thePositronLimiter = new ElectronLimiter(theParSet);
      thePositronLimiter->SetFieldCheckFlag(pLimiter);
      ph->RegisterProcess(theElectronLimiter, G4Positron::Positron());
    }
  }
}
