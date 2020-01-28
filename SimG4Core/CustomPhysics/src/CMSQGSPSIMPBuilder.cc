
#include "SimG4Core/CustomPhysics/interface/CMSQGSPSIMPBuilder.h"
#include "SimG4Core/CustomPhysics/interface/CMSSIMPInelasticProcess.h"

#include "G4SystemOfUnits.hh"
#include "G4ParticleDefinition.hh"
#include "G4TheoFSGenerator.hh"
#include "G4PreCompoundModel.hh"
#include "G4GeneratorPrecompoundInterface.hh"
#include "G4QGSParticipants.hh"
#include "G4QGSMFragmentation.hh"
#include "G4ExcitedStringDecay.hh"

CMSQGSPSIMPBuilder::CMSQGSPSIMPBuilder() {
  theStringModel = new G4QGSModel<G4QGSParticipants>;
  theStringDecay = new G4ExcitedStringDecay(theQGSM = new G4QGSMFragmentation);
  theStringModel->SetFragmentationModel(theStringDecay);
}

CMSQGSPSIMPBuilder::~CMSQGSPSIMPBuilder() {
  delete theStringDecay;
  delete theStringModel;
  delete theQGSM;
}

void CMSQGSPSIMPBuilder::Build(CMSSIMPInelasticProcess* aP) {
  G4GeneratorPrecompoundInterface* theCascade = new G4GeneratorPrecompoundInterface;
  G4PreCompoundModel* thePreEquilib = new G4PreCompoundModel();
  theCascade->SetDeExcitation(thePreEquilib);

  G4TheoFSGenerator* theModel = new G4TheoFSGenerator("QGSP");
  theModel->SetTransport(theCascade);
  theModel->SetHighEnergyGenerator(theStringModel);
  theModel->SetMinEnergy(0.0);
  theModel->SetMaxEnergy(100 * CLHEP::TeV);
  aP->RegisterMe(theModel);
}
