#include "SimG4Core/PhysicsLists/interface/CMSNeutronXS.h"

#include "G4NeutronInelasticXS.hh"
#include "G4NeutronCaptureXS.hh"

#include "G4ParticleDefinition.hh"
#include "G4HadronicProcess.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4HadronicProcessType.hh"

#include "G4Neutron.hh"

CMSNeutronXS::CMSNeutronXS(const G4String&, G4int ver) :
  G4VPhysicsConstructor("Neutron XS"), verbose(ver) {}

CMSNeutronXS::~CMSNeutronXS() {}

void CMSNeutronXS::ConstructParticle() {}

void CMSNeutronXS::ConstructProcess() {

  G4NeutronInelasticXS* xinel = new G4NeutronInelasticXS();
  G4NeutronCaptureXS* xcap = new G4NeutronCaptureXS();

  const G4ParticleDefinition* neutron = G4Neutron::Neutron();
  if(verbose > 1) {
    G4cout << "### CMSNeutronXS: use alternative neutron X-sections"
	   << G4endl;
  }

  G4ProcessVector* pv = neutron->GetProcessManager()->GetProcessList();
  G4int n = pv->size();
  G4HadronicProcess* had = 0;
  for(G4int i=0; i<n; i++) {
    if(fHadronInelastic == ((*pv)[i])->GetProcessSubType()) {
      had = static_cast<G4HadronicProcess*>((*pv)[i]);
      had->AddDataSet(xinel);
    } else if(fCapture == ((*pv)[i])->GetProcessSubType()) {
      had = static_cast<G4HadronicProcess*>((*pv)[i]);
      had->AddDataSet(xcap);
    }
  }
}
