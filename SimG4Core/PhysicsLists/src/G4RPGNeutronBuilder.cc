#include "SimG4Core/PhysicsLists/interface/G4RPGNeutronBuilder.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4ProcessManager.hh"

G4RPGNeutronBuilder::G4RPGNeutronBuilder() : theRPGNeutronModel(0),
					     theNeutronFissionModel(0),
					     theNeutronCaptureModel(0) {
  theMin  = 0;
  theIMin = theMin;
  theMax  = 20*TeV;
  theIMax = 55*GeV;
}

G4RPGNeutronBuilder::~G4RPGNeutronBuilder() {
  if (theNeutronFissionModel) delete theNeutronFissionModel;
  if (theNeutronCaptureModel) delete theNeutronCaptureModel;
  if (theRPGNeutronModel)     delete theRPGNeutronModel;
}

void G4RPGNeutronBuilder::Build(G4HadronElasticProcess *) {
  G4cout << "Info - G4RPGNeutronBuilder::Build() not adding elastic" << G4endl;
}

void G4RPGNeutronBuilder::Build(G4HadronFissionProcess * aP) {
  theNeutronFissionModel = new G4LFission();
  theNeutronFissionModel->SetMinEnergy(theMin);
  theNeutronFissionModel->SetMaxEnergy(theMax);
  aP->RegisterMe(theNeutronFissionModel);
}

void G4RPGNeutronBuilder::Build(G4HadronCaptureProcess * aP) {
  theNeutronCaptureModel = new G4LCapture();
  theNeutronCaptureModel->SetMinEnergy(theMin);
  theNeutronCaptureModel->SetMaxEnergy(theMax);
  aP->RegisterMe(theNeutronCaptureModel);
}

void G4RPGNeutronBuilder::Build(G4NeutronInelasticProcess * aP) {
  if ( theIMax > 1.*eV ) {
    theRPGNeutronModel = new G4RPGNeutronInelastic();
    theRPGNeutronModel->SetMinEnergy(theIMin);
    theRPGNeutronModel->SetMaxEnergy(theIMax);
    aP->RegisterMe(theRPGNeutronModel);
  }
}
