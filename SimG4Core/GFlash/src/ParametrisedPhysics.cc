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
  theHadShowerModel = 0;
  theHadronShowerModel = 0;
}

ParametrisedPhysics::~ParametrisedPhysics() {
  delete theEMShowerModel;
  delete theHadShowerModel;
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

  bool gem  = theParSet.getParameter<bool>("GflashEMShowerModel");
  bool ghad = theParSet.getParameter<bool>("GflashHadronShowerModel");
  std::cout << "GFlash Construct: " << gem << "  " << ghad << std::endl;

  if(gem) {
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

    G4Region* aRegion = G4RegionStore::GetInstance()->GetRegion("EcalRegion");
    //  G4Region* aRegion = G4RegionStore::GetInstance()->GetRegion("CaloRegion");
    if(!aRegion){
      std::cout << "EcalRegion is not defined !!!" << std::endl;
      std::cout << "This means that GFlash will not be turned on." << std::endl;
      std::cout << "Take a look at cmsGflashGeometryXML.cfi if it includes gflashCaloProdCuts.xml." << std::endl;
	
    } else {

      //Electromagnetic Shower Model
      theEMShowerModel = 
	new GflashEMShowerModel("GflashEMShowerModel",aRegion,theParSet);
      std::cout << "GFlash is defined for EcalRegion" << std::endl;
    }    
    /*
    aRegion = G4RegionStore::GetInstance()->GetRegion("HcalRegion");
    //  G4Region* aRegion = G4RegionStore::GetInstance()->GetRegion("CaloRegion");
    if(!aRegion){
      std::cout << "CaloRegion is not defined !!!" << std::endl;
      std::cout << "This means that GFlash will not be turned on." << std::endl;
      std::cout << "Take a look at cmsGflashGeometryXML.cfi if it includes gflashCaloProdCuts.xml." << std::endl;
	
    } else {
      //Hadronic Shower Model
      //if(theParSet.getParameter<bool>("GflashHadronShowerModel")) {
      theHadShowerModel = 
	new GflashEMShowerModel("GflashHadShowerModel",aRegion,theParSet);
      //new GflashHadronShowerModel("GflashHadronShowerModel",aRegion,theParSet);
	std::cout << "GFlash is defined for HcalRegion" << std::endl;
    }
    */
  }
}
