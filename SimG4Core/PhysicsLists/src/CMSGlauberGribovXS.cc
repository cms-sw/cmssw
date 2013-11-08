#include "SimG4Core/PhysicsLists/interface/CMSGlauberGribovXS.h"
#include "G4ParticleDefinition.hh"
#include "G4HadronicProcess.hh"
#include "G4GlauberGribovCrossSection.hh"
#include "G4BGGNucleonInelasticXS.hh"
#include "G4BGGPionInelasticXS.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4HadronicProcessType.hh"

#include "G4PionPlus.hh"
#include "G4PionMinus.hh"
#include "G4KaonPlus.hh"
#include "G4KaonMinus.hh"
#include "G4BMesonMinus.hh"
#include "G4BMesonPlus.hh"
#include "G4DMesonMinus.hh"
#include "G4DMesonPlus.hh"
#include "G4Proton.hh"
#include "G4AntiProton.hh"
#include "G4SigmaMinus.hh"
#include "G4AntiSigmaMinus.hh"
#include "G4SigmaPlus.hh"
#include "G4AntiSigmaPlus.hh"
#include "G4XiMinus.hh"
#include "G4AntiXiMinus.hh"
#include "G4OmegaMinus.hh"
#include "G4AntiOmegaMinus.hh"
#include "G4LambdacPlus.hh"
#include "G4AntiLambdacPlus.hh"
#include "G4XicPlus.hh"
#include "G4AntiXicPlus.hh"
#include "G4Deuteron.hh"
#include "G4Triton.hh"
#include "G4He3.hh"
#include "G4Alpha.hh"
#include "G4GenericIon.hh"

#include "G4SystemOfUnits.hh"

CMSGlauberGribovXS::CMSGlauberGribovXS(G4int ver) :
  G4VPhysicsConstructor("GlauberGribov XS"), verbose(ver) 
{}

CMSGlauberGribovXS::~CMSGlauberGribovXS() {}

void CMSGlauberGribovXS::ConstructParticle() {}

void CMSGlauberGribovXS::ConstructProcess() 
{

  G4GlauberGribovCrossSection* gg = new G4GlauberGribovCrossSection();
  gg->SetEnergyLowerLimit(90.*GeV);

  aParticleIterator->reset();
  while( (*aParticleIterator)() ){
    G4ParticleDefinition* particle = aParticleIterator->value();
    G4String particleName = particle->GetParticleName();
    if(verbose > 1) {
      G4cout << "### " << GetPhysicsName() << " instantiates for " 
	     << particleName << G4endl;
    }

    if (particleName == "neutron" ||
	particleName == "pi+" ||
	particleName == "pi-" ||
	particleName == "proton") {

      G4ProcessVector*  pv = particle->GetProcessManager()->GetProcessList();
      G4int n = pv->size();
      G4HadronicProcess* had = 0;
      for(G4int i=0; i<n; i++) {
        if(fHadronInelastic == ((*pv)[i])->GetProcessSubType()) {
          had = static_cast<G4HadronicProcess*>((*pv)[i]);
          break;
	}
      }
	
      if(verbose > 0) {
	G4cout << "### CMSGlauberGribovXS::ConstructProcess for " << particleName;
      }
      if(had) {
	if(verbose > 0) {G4cout << " and  " << had->GetProcessName();}   
	if (particleName == "neutron") {
	  had->AddDataSet(new G4BGGNucleonInelasticXS(particle));
	} else if(particleName == "proton") {
	  had->AddDataSet(new G4BGGNucleonInelasticXS(particle));
	} else if(particleName == "pi+") {
	  had->AddDataSet(new G4BGGPionInelasticXS(particle));
	} else if(particleName == "pi-") {
	  had->AddDataSet(new G4BGGPionInelasticXS(particle));
	}
	   //	had->AddDataSet(gg);
      }
      if(verbose > 0) {G4cout << G4endl;}   
    }
  }
}
