#include "SimG4Core/CustomPhysics/interface/CustomPhysicsList.h"
#include "SimG4Core/CustomPhysics/interface/CustomParticleFactory.h"
#include "SimG4Core/CustomPhysics/interface/DummyChargeFlipProcess.h"
#include "SimG4Core/CustomPhysics/interface/G4ProcessHelper.hh"
#include "SimG4Core/CustomPhysics/interface/CustomPDGParser.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4Decay.hh"
#include "G4hMultipleScattering.hh"
#include "G4hIonisation.hh"
#include "G4ProcessManager.hh"

#include "SimG4Core/CustomPhysics/interface/FullModelHadronicProcess.hh"
#include "SimG4Core/CustomPhysics/interface/ToyModelHadronicProcess.hh"

using namespace CLHEP;

G4ThreadLocal G4Decay* CustomPhysicsList::fDecayProcess = 0;
G4ThreadLocal G4ProcessHelper* CustomPhysicsList::myHelper = 0;
G4ThreadLocal bool CustomPhysicsList::fInitialized = false;

CustomPhysicsList::CustomPhysicsList(std::string name, const edm::ParameterSet & p)  
  :  G4VPhysicsConstructor(name) 
{  
  myConfig = p;
  edm::FileInPath fp = p.getParameter<edm::FileInPath>("particlesDef");
  fHadronicInteraction = p.getParameter<bool>("rhadronPhysics");

  particleDefFilePath = fp.fullPath();
  edm::LogInfo("SimG4CoreCustomPhysics") 
    << "CustomPhysicsList: Path for custom particle definition file: \n"
    <<particleDefFilePath;
}

CustomPhysicsList::~CustomPhysicsList() {
}
 
void CustomPhysicsList::ConstructParticle(){
  G4cout << "===== CustomPhysicsList::ConstructParticle " << this << G4endl;
  CustomParticleFactory::loadCustomParticles(particleDefFilePath);
}
 
void CustomPhysicsList::ConstructProcess() {
  
  //if(fInitialized) { return; }
  //fInitialized = true;

  edm::LogInfo("SimG4CoreCustomPhysics") 
    <<"CustomPhysicsList: adding CustomPhysics processes "
    << "for the list of particles";

  fDecayProcess = new G4Decay();

  aParticleIterator->reset();

  while((*aParticleIterator)()) {
    G4ParticleDefinition* particle = aParticleIterator->value();
    if(CustomParticleFactory::isCustomParticle(particle)) {
      CustomParticle* cp = dynamic_cast<CustomParticle*>(particle);
      G4ProcessManager* pmanager = particle->GetProcessManager();
      edm::LogInfo("SimG4CoreCustomPhysics") 
	<<"CustomPhysicsList: " << particle->GetParticleName()
	<<"  PDGcode= " << particle->GetPDGEncoding()
	<< "  Mass= " << particle->GetPDGMass()/GeV  <<" GeV.";
      if(cp && pmanager) {
	if(particle->GetPDGCharge() != 0.0) {
	  pmanager->AddProcess(new G4hMultipleScattering,-1, 1, 1);
	  pmanager->AddProcess(new G4hIonisation,        -1, 2, 2);
	}
	if(fDecayProcess->IsApplicable(*particle)) {
	  pmanager->AddProcess(new G4Decay, 0, -1, 3);
	}
	if(cp->GetCloud() && fHadronicInteraction && 
	   CustomPDGParser::s_isRHadron(particle->GetPDGEncoding())) {
	  edm::LogInfo("SimG4CoreCustomPhysics") 
	    <<"CustomPhysicsList: " << particle->GetParticleName()
	    <<" CloudMass= " <<cp->GetCloud()->GetPDGMass()/GeV
	    <<" GeV; SpectatorMass= " << cp->GetSpectator()->GetPDGMass()/GeV<<" GeV.";
       
	  if(!myHelper) myHelper = new G4ProcessHelper(myConfig);
	  pmanager->AddDiscreteProcess(new FullModelHadronicProcess(myHelper));
	}
      }
    }
  }
}
