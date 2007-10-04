#include "SimG4Core/GFlash/interface/ParametrisedPhysics.h"

#include "G4Electron.hh"
#include "G4FastSimulationManagerProcess.hh"
#include "G4ProcessManager.hh"

#include "G4LeptonConstructor.hh"
#include "G4MesonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4ShortLivedConstructor.hh"
#include "G4IonConstructor.hh"

ParametrisedPhysics::ParametrisedPhysics(std::string name) : G4VPhysicsConstructor(name) {}

ParametrisedPhysics::~ParametrisedPhysics() {}

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

void ParametrisedPhysics::ConstructProcess() { addParametrisation(); }

void ParametrisedPhysics::addParametrisation()
{
    std::cout << " ParametrisedPhysics: adding the FastSimulationManagerProcess " << std::endl;
    G4FastSimulationManagerProcess * theFastSimulationManagerProcess = 
	new G4FastSimulationManagerProcess();
    theParticleIterator->reset();
    while ((*theParticleIterator)())
    {
	G4ParticleDefinition * particle = theParticleIterator->value();
	G4ProcessManager * pmanager = particle->GetProcessManager();
	pmanager->AddProcess(theFastSimulationManagerProcess, -1, -1, 1);
    }
}
