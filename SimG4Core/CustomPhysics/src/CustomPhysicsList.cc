#include "SimG4Core/CustomPhysics/interface/CustomPhysicsList.h"
#include "SimG4Core/CustomPhysics/interface/CustomParticleFactory.h"
#include "SimG4Core/CustomPhysics/interface/DummyChargeFlipProcess.h"
#include "SimG4Core/CustomPhysics/interface/GenericHadronicProcess.h"

#include "G4MultipleScattering.hh"
#include "G4hIonisation.hh"
#include "G4MuIonisation.hh"
#include "G4ProcessManager.hh"

#include "G4LeptonConstructor.hh"
#include "G4MesonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4ShortLivedConstructor.hh"
#include "G4IonConstructor.hh"
 
CustomPhysicsList::CustomPhysicsList(std::string name,const edm::ParameterSet & p) : 
    G4VPhysicsConstructor(name), m_pCustomPhysicsList(p) {}

CustomPhysicsList::~CustomPhysicsList() {}
 
void CustomPhysicsList::ConstructParticle()
{
    CustomParticleFactory::loadCustomParticles();     

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
 
void CustomPhysicsList::ConstructProcess() { addCustomPhysicsList(); }
 
void CustomPhysicsList::addCustomPhysicsList()
{
    std::cout << " CustomPhysicsList: adding CustomPhysicsList processes  " 
	      << std::endl;
    theParticleIterator->reset();
    while ((*theParticleIterator)())
    {	
	G4ParticleDefinition * particle = theParticleIterator->value();
	if (CustomParticleFactory::isCustomParticle(particle))
	{
	    std::cout << particle->GetParticleName() 
		      << " is Custom" << std::endl;
	    G4ProcessManager * pmanager = particle->GetProcessManager();
	    if (pmanager!=0)
	    { 
		int i=1;
		if (particle->GetParticleType()=="rhadron")  
		{
		    if (m_pCustomPhysicsList.getParameter<bool>("RHadronDummyFlip"))
			pmanager->AddProcess(new DummyChargeFlipProcess,-1, -1,i++);
                    else
			pmanager->AddProcess(new GenericHadronicProcess,-1, -1,i++);
                
                }
		if (particle->GetPDGCharge()/eplus != 0)
		{ 
		    pmanager->AddProcess(new G4MultipleScattering,-1, 1,i++);
		    pmanager->AddProcess(new G4hIonisation,       -1, 2,i++);
		}
                else std::cout << "   It is neutral!!" << std::endl;
            }
	    else std::cout << "   No pmanager " << std::endl;
	}
    }
}
