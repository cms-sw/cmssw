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
  G4VPhysicsConstructor(name), theParSet(p) 
{
  theEMShowerModel = 0;
  theHadronShowerModel = 0;
}

ParametrisedPhysics::~ParametrisedPhysics() {
  //if(theParSet.getParameter<bool>("GflashEMShowerModel") && theEMShowerModel) 
    delete theEMShowerModel;
    //if(theParSet.getParameter<bool>("GflashHadronShowerModel") && theHadronShowerModel) 
    delete theHadronShowerModel;
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

  G4FastSimulationManagerProcess * theFastSimulationManagerProcess = new G4FastSimulationManagerProcess();
  theParticleIterator->reset();
  while ((*theParticleIterator)()) {
    G4ParticleDefinition * particle = theParticleIterator->value();
    G4ProcessManager * pmanager = particle->GetProcessManager();
    G4String pname = particle->GetParticleName();
    if(pname == "e-" || pname == "e+") {
      pmanager->AddProcess(theFastSimulationManagerProcess, -1, -1, 1);
    }
  }

  // GflashEnvelop definition as CaloRegion which includes EcalRegion & HcalRegion
  G4Region* aRegion = G4RegionStore::GetInstance()->GetRegion("EcalRegion");
  //  G4Region* aRegion = G4RegionStore::GetInstance()->GetRegion("CaloRegion");
  if(aRegion == 0){
    std::cout << "CaloRegion is not defined !!!" << std::endl;
    std::cout << "This means that GFlash will not be turned on." << std::endl;
    std::cout << "Take a look at cmsGflashGeometryXML.cfi if it includes gflashCaloProdCuts.xml." << std::endl;
    
  }

  //Electromagnetic Shower Model
  if(theParSet.getParameter<bool>("GflashEMShowerModel")) {
    theEMShowerModel = 
      new GflashEMShowerModel("GflashEMShowerModel",aRegion,theParSet);
  }    

  aRegion = G4RegionStore::GetInstance()->GetRegion("HcalRegion");
  //  G4Region* aRegion = G4RegionStore::GetInstance()->GetRegion("CaloRegion");
  if(aRegion == 0){
    std::cout << "CaloRegion is not defined !!!" << std::endl;
    std::cout << "This means that GFlash will not be turned on." << std::endl;
    std::cout << "Take a look at cmsGflashGeometryXML.cfi if it includes gflashCaloProdCuts.xml." << std::endl;
    
  }
  //Hadronic Shower Model
  if(theParSet.getParameter<bool>("GflashHadronShowerModel")) {
    theHadronShowerModel = 
      new GflashHadronShowerModel("GflashHadronShowerModel",aRegion,theParSet);
  }
}
