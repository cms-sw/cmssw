//
// CHIPS for sampling scattering for p and n
// Glauber model for samplimg of high energy pi+- (E > 1GeV)
// LHEP sampling model for the other particle
// BBG cross sections for p and pi+- 
// XS cross sections for neutrons
// LHEP cross sections for other particles

#include "SimG4Core/PhysicsLists/interface/CMSHadronElasticPhysicsXS.h"

#include "G4ParticleDefinition.hh"
#include "G4ProcessManager.hh"

#include "G4MesonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4IonConstructor.hh"
#include "G4Neutron.hh"

#include "G4WHadronElasticProcess.hh"
//#include "G4VHadronElastic.hh"
#include "G4CHIPSElastic.hh"
#include "G4ElasticHadrNucleusHE.hh"
#include "G4BGGNucleonElasticXS.hh"
#include "G4BGGPionElasticXS.hh"
#include "G4NeutronElasticXS.hh"

CMSHadronElasticPhysicsXS::CMSHadronElasticPhysicsXS(G4int ver)
  : G4VPhysicsConstructor("hElasticWEL_CHIPS_XS"), verbose(ver), 
    wasActivated(false)
{
  if(verbose > 1) { 
    G4cout << "### G4HadronElasticPhysicsHP: " << GetPhysicsName() 
	   << G4endl; 
  }
}

CMSHadronElasticPhysicsXS::~CMSHadronElasticPhysicsXS()
{}

void CMSHadronElasticPhysicsXS::ConstructParticle()
{
  // G4cout << "G4HadronElasticPhysics::ConstructParticle" << G4endl;
  G4MesonConstructor pMesonConstructor;
  pMesonConstructor.ConstructParticle();

  G4BaryonConstructor pBaryonConstructor;
  pBaryonConstructor.ConstructParticle();

  //  Construct light ions
  G4IonConstructor pConstructor;
  pConstructor.ConstructParticle();  
}

void CMSHadronElasticPhysicsXS::ConstructProcess()
{
  if(wasActivated) return;
  wasActivated = true;

  G4double elimit = 1.0*GeV;
  if(verbose > 1) {
    G4cout << "### HadronElasticPhysics Construct Processes with HE limit " 
	   << elimit << " MeV" << G4endl;
  }

  //G4VHadronElastic* plep0 = new G4VHadronElastic();
  //G4VHadronElastic* plep1 = new G4VHadronElastic();
  G4HadronElastic* plep0 = new G4HadronElastic();
  G4HadronElastic* plep1 = new G4HadronElastic();
  plep1->SetMaxEnergy(elimit);

  G4CHIPSElastic* chipsp = new G4CHIPSElastic();
  G4CHIPSElastic* chipsn = new G4CHIPSElastic();

  G4ElasticHadrNucleusHE* he = new G4ElasticHadrNucleusHE(); 
  he->SetMinEnergy(elimit);

  theParticleIterator->reset();
  while( (*theParticleIterator)() )
  {
    G4ParticleDefinition* particle = theParticleIterator->value();
    G4String pname = particle->GetParticleName();
    if(pname == "anti_lambda"  ||
       pname == "anti_neutron" ||
       pname == "anti_omega-"  || 
       pname == "anti_proton"  || 
       pname == "anti_sigma-"  || 
       pname == "anti_sigma+"  || 
       pname == "anti_xi-"  || 
       pname == "anti_xi0"  || 
       pname == "kaon-"     || 
       pname == "kaon+"     || 
       pname == "kaon0S"    || 
       pname == "kaon0L"    || 
       pname == "lambda"    || 
       pname == "omega-"    || 
       pname == "pi-"       || 
       pname == "pi+"       || 
       pname == "proton"    || 
       pname == "sigma-"    || 
       pname == "sigma+"    || 
       pname == "xi-"       || 
       pname == "alpha"     ||
       pname == "deuteron"  ||
       pname == "triton") {
      
      G4ProcessManager* pmanager = particle->GetProcessManager();
      G4WHadronElasticProcess* hel = new G4WHadronElasticProcess();
      if(pname == "proton") { 
	hel->AddDataSet(new G4BGGNucleonElasticXS(particle));
	hel->RegisterMe(chipsp);
      } else if (pname == "pi+" || pname == "pi-") { 
	hel->AddDataSet(new G4BGGPionElasticXS(particle));
	hel->RegisterMe(plep1);
	hel->RegisterMe(he);
      } else {
	hel->RegisterMe(plep0);
      }
      pmanager->AddDiscreteProcess(hel);
      if(verbose > 1) {
	G4cout << "### HadronElasticPhysicsXS: " << hel->GetProcessName()
	       << " added for " << particle->GetParticleName() << G4endl;
      }

      // neutron case
    } else if(pname == "neutron") {   

      G4ProcessManager* pmanager = particle->GetProcessManager();
      G4WHadronElasticProcess* hel = new G4WHadronElasticProcess();
      hel->AddDataSet(new G4NeutronElasticXS());
      hel->RegisterMe(chipsn);

      pmanager->AddDiscreteProcess(hel);

      if(verbose > 1) {
	G4cout << "### HadronElasticPhysicsXS: " << hel->GetProcessName()
	       << " added for " << particle->GetParticleName() << G4endl;
      }
    }
  }
}


