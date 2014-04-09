#include "SimG4Core/CustomPhysics/interface/CustomPhysicsList.h"
#include "SimG4Core/CustomPhysics/interface/CustomParticleFactory.h"
#include "SimG4Core/CustomPhysics/interface/DummyChargeFlipProcess.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4Decay.hh"
#include "G4hMultipleScattering.hh"
#include "G4hIonisation.hh"
#include "G4ProcessManager.hh"

#include "G4LeptonConstructor.hh"
#include "G4MesonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4ShortLivedConstructor.hh"
#include "G4IonConstructor.hh"

#include "SimG4Core/CustomPhysics/interface/FullModelHadronicProcess.hh"
#include "SimG4Core/CustomPhysics/interface/ToyModelHadronicProcess.hh"

using namespace CLHEP;
 

CustomPhysicsList::CustomPhysicsList(std::string name, const edm::ParameterSet & p)  :  G4VPhysicsConstructor(name) {
  
  myConfig = p;
  edm::FileInPath fp = p.getParameter<edm::FileInPath>("particlesDef");
  particleDefFilePath = fp.fullPath();
  edm::LogInfo("CustomPhysics")<<"Path for custom particle definition file: "
			       <<particleDefFilePath;
  myHelper = 0;
  
 }

CustomPhysicsList::~CustomPhysicsList() {
  delete myHelper;
}
 
void CustomPhysicsList::ConstructParticle(){
  CustomParticleFactory::loadCustomParticles(particleDefFilePath);     
}
 
void CustomPhysicsList::ConstructProcess() {
  addCustomPhysics();
}
 
void CustomPhysicsList::addCustomPhysics(){
  LogDebug("CustomPhysics") << " CustomPhysics: adding CustomPhysics processes";
  aParticleIterator->reset();

  while((*aParticleIterator)())    {
    int i = 0;
    G4ParticleDefinition* particle = aParticleIterator->value();
    CustomParticle* cp = dynamic_cast<CustomParticle*>(particle);
    if(CustomParticleFactory::isCustomParticle(particle)) {
      LogDebug("CustomPhysics") << particle->GetParticleName()
				<<", "<<particle->GetPDGEncoding()
				<< " is Custom. Mass is "
				<<particle->GetPDGMass()/GeV  <<" GeV.";
      if(cp->GetCloud()!=0) {
	LogDebug("CustomPhysics")<<"Cloud mass is "
				 <<cp->GetCloud()->GetPDGMass()/GeV
				 <<" GeV. Spectator mass is "
				 <<static_cast<CustomParticle*>(particle)->GetSpectator()->GetPDGMass()/GeV
				 <<" GeV.";
      }
      G4ProcessManager* pmanager = particle->GetProcessManager();
      if(pmanager) {
	if(cp!=0) {
	  if(particle->GetParticleType()=="rhadron" || 
	     particle->GetParticleType()=="mesonino" || 
	     particle->GetParticleType() == "sbaryon"){
	    if(!myHelper) myHelper = new G4ProcessHelper(myConfig);
	    pmanager->AddDiscreteProcess(new FullModelHadronicProcess(myHelper));
	  }
	}
	if(particle->GetPDGCharge()/eplus != 0) {
	  pmanager->AddProcess(new G4hMultipleScattering,-1, 1,i+1);
	  pmanager->AddProcess(new G4hIonisation,        -1, 2,i+2);
	}
      }
      else      LogDebug("CustomPhysics") << "   No pmanager";
    }
  }
}


void CustomPhysicsList::setupRHadronPhycis(G4ParticleDefinition* particle){

  //    LogDebug("CustomPhysics")<<"Configuring rHadron: "
  //	<<cp->

  CustomParticle* cp = dynamic_cast<CustomParticle*>(particle);
  if(cp->GetCloud()!=0) 
    LogDebug("CustomPhysics")<<"Cloud mass is "
			     <<cp->GetCloud()->GetPDGMass()/GeV
			     <<" GeV. Spectator mass is "
			     <<static_cast<CustomParticle*>(particle)->GetSpectator()->GetPDGMass()/GeV
			     <<" GeV.";
  
  G4ProcessManager* pmanager = particle->GetProcessManager();
  if(pmanager){
    if(!myHelper) myHelper = new G4ProcessHelper(myConfig);
    pmanager->AddDiscreteProcess(new FullModelHadronicProcess(myHelper)); //GHEISHA
    if(particle->GetPDGCharge()/eplus != 0){
      pmanager->AddProcess(new G4hMultipleScattering,-1, 1,1);
      pmanager->AddProcess(new G4hIonisation,        -1, 2,2);
    }
  }
  else      LogDebug("CustomPhysics") << "   No pmanager";
}
					       

void CustomPhysicsList::setupSUSYPhycis(G4ParticleDefinition* particle){

//  CustomParticle* cp = dynamic_cast<CustomParticle*>(particle);
  G4ProcessManager* pmanager = particle->GetProcessManager();
  if(pmanager){
    pmanager->AddProcess(new G4Decay,1, 1,1);
    if(particle->GetPDGCharge()/eplus != 0){
      pmanager->AddProcess(new G4hMultipleScattering,-1, 2,2);
      pmanager->AddProcess(new G4hIonisation,        -1, 3,3);
    }
  }
  else      LogDebug("CustomPhysics") << "   No pmanager";
}
