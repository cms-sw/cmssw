#include "SimG4Core/PhysicsLists/interface/G4RPGProtonBuilder.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4ProcessManager.hh"

G4RPGProtonBuilder::G4RPGProtonBuilder(): theRPGProtonModel(0) {
  theMin = 0;
  theMax = 55*GeV;
}

G4RPGProtonBuilder::~G4RPGProtonBuilder() {
  if (theRPGProtonModel) delete theRPGProtonModel;
}

void G4RPGProtonBuilder::Build(G4HadronElasticProcess *) {
  G4cout << "Info - G4RPGProtonBuilder::Build() not adding elastic" << G4endl;
}

void G4RPGProtonBuilder::Build(G4ProtonInelasticProcess * aP) {
  theRPGProtonModel = new G4RPGProtonInelastic();
  theRPGProtonModel->SetMinEnergy(theMin);
  theRPGProtonModel->SetMaxEnergy(theMax);
  aP->RegisterMe(theRPGProtonModel);
}
