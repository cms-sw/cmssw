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


ParametrisedPhysics::ParametrisedPhysics(std::string name, const edm::ParameterSet & p) :
  G4VPhysicsConstructor(name), theParSet(p) {}

ParametrisedPhysics::~ParametrisedPhysics() {
  if(theEMShowerModel)     delete theEMShowerModel;
  if(theHadronShowerModel) delete theHadronShowerModel;
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

  std::cout << " ParametrisedPhysics: adding the FastSimulationManagerProcess " << std::endl;
  G4FastSimulationManagerProcess * theFastSimulationManagerProcess = new G4FastSimulationManagerProcess();
  theParticleIterator->reset();
  while ((*theParticleIterator)()) {
    G4ParticleDefinition * particle = theParticleIterator->value();
    G4ProcessManager * pmanager = particle->GetProcessManager();
    pmanager->AddProcess(theFastSimulationManagerProcess, -1, -1, 1);
  }

  G4Region* aRegion = G4RegionStore::GetInstance()->GetRegion("DefaultRegionForTheWorld");

  //Electromagnetic Shower Model
  if(theParSet.getParameter<bool>("GflashEMShowerModel")) {
    theEMShowerModel  = new GflashEMShowerModel("GflashEMShowerModel",aRegion);
  }    

  //Hadronic Shower Model
  if(theParSet.getParameter<bool>("GflashHadronShowerModel")) {
    theHadronShowerModel = new GflashHadronShowerModel("GflashHadronShowerModel",aRegion);
  }


}
