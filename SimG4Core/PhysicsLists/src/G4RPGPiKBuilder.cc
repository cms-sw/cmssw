#include "SimG4Core/PhysicsLists/interface/G4RPGPiKBuilder.hh"

#include "globals.hh"
#include "G4ios.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4MesonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4ProcessManager.hh"

G4RPGPiKBuilder::G4RPGPiKBuilder(): theRPGPiPlusModel(0),theRPGPiMinusModel(0),
				    theRPGKPlusModel(0), theRPGKMinusModel(0),
				    theRPGKLongModel(0), theRPGKShortModel(0) {
  theMin = 0;
  theMax = 55*GeV;
}

G4RPGPiKBuilder::~G4RPGPiKBuilder() {
  if (theRPGPiPlusModel)  delete theRPGPiPlusModel;
  if (theRPGPiMinusModel) delete theRPGPiMinusModel;
  if (theRPGKPlusModel)   delete theRPGKPlusModel;
  if (theRPGKMinusModel)  delete theRPGKMinusModel;
  if (theRPGKLongModel)   delete theRPGKShortModel;
  if (theRPGKShortModel)  delete theRPGKLongModel;
}

void G4RPGPiKBuilder::Build(G4HadronElasticProcess *) {
  G4cout << "Info - G4RPGPiKBuilder::Build() not adding elastic" << G4endl;
}

void G4RPGPiKBuilder::Build(G4PionPlusInelasticProcess * aP) {
  theRPGPiPlusModel = new G4RPGPiPlusInelastic();
  theRPGPiPlusModel->SetMinEnergy(theMin);
  theRPGPiPlusModel->SetMaxEnergy(theMax);
  aP->RegisterMe(theRPGPiPlusModel);
}

void G4RPGPiKBuilder::Build(G4PionMinusInelasticProcess * aP) {
  theRPGPiMinusModel = new G4RPGPiMinusInelastic();
  theRPGPiMinusModel->SetMinEnergy(theMin);
  theRPGPiMinusModel->SetMaxEnergy(theMax);
  aP->RegisterMe(theRPGPiMinusModel);
}

void G4RPGPiKBuilder::Build(G4KaonPlusInelasticProcess * aP) {
  theRPGKPlusModel = new G4RPGKPlusInelastic();
  theRPGKPlusModel->SetMinEnergy(theMin);
  theRPGKPlusModel->SetMaxEnergy(theMax);
  aP->RegisterMe(theRPGKPlusModel);
}

void G4RPGPiKBuilder::Build(G4KaonMinusInelasticProcess * aP) {
  theRPGKMinusModel = new G4RPGKMinusInelastic();
  theRPGKMinusModel->SetMaxEnergy(theMax);
  theRPGKMinusModel->SetMinEnergy(theMin);
  aP->RegisterMe(theRPGKMinusModel);
}

void G4RPGPiKBuilder::Build(G4KaonZeroLInelasticProcess * aP) {
  theRPGKLongModel = new G4RPGKLongInelastic();
  theRPGKLongModel->SetMaxEnergy(theMax);
  theRPGKLongModel->SetMinEnergy(theMin);
  aP->RegisterMe(theRPGKLongModel);
}
 
void G4RPGPiKBuilder::Build(G4KaonZeroSInelasticProcess * aP) {
  theRPGKShortModel = new G4RPGKShortInelastic();
  theRPGKShortModel->SetMaxEnergy(theMax);
  theRPGKShortModel->SetMinEnergy(theMin);
  aP->RegisterMe(theRPGKShortModel);
}
