#include "SimG4Core/CustomPhysics/interface/CustomPhysicsListSS.h"
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
#include "G4CoulombScattering.hh"
#include "G4ProcessManager.hh"

#include "SimG4Core/CustomPhysics/interface/FullModelHadronicProcess.hh"
#include "SimG4Core/CustomPhysics/interface/ToyModelHadronicProcess.hh"

using namespace CLHEP;

G4ThreadLocal G4Decay* CustomPhysicsListSS::fDecayProcess = nullptr;
G4ThreadLocal G4ProcessHelper* CustomPhysicsListSS::myHelper = nullptr;
 
CustomPhysicsListSS::CustomPhysicsListSS(std::string name, const edm::ParameterSet& p)
  :  G4VPhysicsConstructor(name) 
{  
  myConfig = p;
  edm::FileInPath fp = p.getParameter<edm::FileInPath>("particlesDef");
  fHadronicInteraction = p.getParameter<bool>("rhadronPhysics");
  particleDefFilePath = fp.fullPath();
  edm::LogInfo("SimG4CoreCustomPhysics")
    <<"CustomPhysicsListSS: Path for custom particle definition file: \n" 
    <<particleDefFilePath;
  myHelper = nullptr;  
}

CustomPhysicsListSS::~CustomPhysicsListSS() {
}
 
void CustomPhysicsListSS::ConstructParticle(){
  CustomParticleFactory::loadCustomParticles(particleDefFilePath);     
}
 
void CustomPhysicsListSS::ConstructProcess() {

  edm::LogInfo("SimG4CoreCustomPhysics") 
    <<"CustomPhysicsListSS: adding CustomPhysics processes";

  fDecayProcess = new G4Decay();
  G4PhysicsListHelper* ph = G4PhysicsListHelper::GetPhysicsListHelper();

  for(auto particle : CustomParticleFactory::GetCustomParticles()) {

    CustomParticle* cp = dynamic_cast<CustomParticle*>(particle);
    if(cp) {
      G4ProcessManager* pmanager = particle->GetProcessManager();
      edm::LogInfo("SimG4CoreCustomPhysics") 
	<<"CustomPhysicsListSS: " << particle->GetParticleName()
	<<" PDGcode= " << particle->GetPDGEncoding()
	<< " Mass= " << particle->GetPDGMass()/GeV  <<" GeV.";
      if(pmanager) {
	if(particle->GetPDGCharge() != 0.0) {
	  ph->RegisterProcess(new G4hMultipleScattering, particle);
	  ph->RegisterProcess(new G4hIonisation, particle);
	}
	if(fDecayProcess->IsApplicable(*particle)) {
	  ph->RegisterProcess(fDecayProcess, particle);
	}
	if(cp->GetCloud() && fHadronicInteraction &&
	   CustomPDGParser::s_isRHadron(particle->GetPDGEncoding())) {
	  edm::LogInfo("SimG4CoreCustomPhysics") 
	    <<"CustomPhysicsListSS: " << particle->GetParticleName()
	    <<" CloudMass= " <<cp->GetCloud()->GetPDGMass()/GeV
	    <<" GeV; SpectatorMass= " << cp->GetSpectator()->GetPDGMass()/GeV <<" GeV.";
       
	  if(!myHelper) { myHelper = new G4ProcessHelper(myConfig); }
	  pmanager->AddDiscreteProcess(new FullModelHadronicProcess(myHelper));
	}
      }
    }
  }
}
