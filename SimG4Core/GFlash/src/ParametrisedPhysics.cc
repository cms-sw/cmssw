#include "SimG4Core/GFlash/interface/ParametrisedPhysics.h"

#include "G4Electron.hh"
#include "G4FastSimulationManagerProcess.hh"
#include "G4ProcessManager.hh"

#include "G4LeptonConstructor.hh"
#include "G4MesonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4ShortLivedConstructor.hh"
#include "G4IonConstructor.hh"
#include "G4RegionStore.hh"

using namespace CLHEP;

G4ThreadLocal ParametrisedPhysics::ThreadPrivate* ParametrisedPhysics::tpdata = nullptr;

ParametrisedPhysics::ParametrisedPhysics(std::string name, const edm::ParameterSet & p) :
  G4VPhysicsConstructor(name), theParSet(p) 
{}

ParametrisedPhysics::~ParametrisedPhysics() {
  if(nullptr != tpdata) {
    delete tpdata->theEMShowerModel;
    delete tpdata->theHadShowerModel;
    delete tpdata->theHadronShowerModel;
    delete tpdata;
    tpdata = nullptr;
  }
}

void ParametrisedPhysics::ConstructParticle() 
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

void ParametrisedPhysics::ConstructProcess() {

  tpdata = new ThreadPrivate;
  tpdata->theEMShowerModel = nullptr;
  tpdata->theHadShowerModel = nullptr;
  tpdata->theHadronShowerModel = nullptr;

  bool gem  = theParSet.getParameter<bool>("GflashEcal");
  bool ghad = theParSet.getParameter<bool>("GflashHcal");
  G4cout << "GFlash Construct: " << gem << "  " << ghad << G4endl;

  if(gem || ghad) {
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
	G4cout << "EcalRegion is not defined !!!" << G4endl;
	G4cout << "This means that GFlash will not be turned on." << G4endl;
	
      } else {

	//Electromagnetic Shower Model for ECAL
	tpdata->theEMShowerModel = 
	  new GflashEMShowerModel("GflashEMShowerModel",aRegion,theParSet);
	G4cout << "GFlash is defined for EcalRegion" << G4endl;
      }    
    }
    if(ghad) {
      G4Region* aRegion = G4RegionStore::GetInstance()->GetRegion("HcalRegion");
      if(!aRegion) {
	G4cout << "HcalRegion is not defined !!!" << G4endl;
	G4cout << "This means that GFlash will not be turned on." << G4endl;
	
      } else {

	//Electromagnetic Shower Model for HCAL
	tpdata->theHadShowerModel = 
	  new GflashEMShowerModel("GflashHadShowerModel",aRegion,theParSet);
	G4cout << "GFlash is defined for HcalRegion" << G4endl;    
      }
    }
  }
}
