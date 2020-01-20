
#include "SimG4Core/CustomPhysics/interface/CMSQGSPSIMPBuilder.h"
#include "G4SystemOfUnits.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4ProcessManager.hh"
#include "G4ExcitationHandler.hh"

#include "SimG4Core/CustomPhysics/interface/CMSSIMPInelasticXS.h"
#include "SimG4Core/CustomPhysics/interface/CMSSIMP.h"

CMSQGSPSIMPBuilder::CMSQGSPSIMPBuilder(G4bool quasiElastic) {
  theMin = 12 * GeV;

  theModel = new G4TheoFSGenerator("QGSP");

  theStringModel = new G4QGSModel<G4QGSParticipants>;
  theStringDecay = new G4ExcitedStringDecay(theQGSM = new G4QGSMFragmentation);
  theStringModel->SetFragmentationModel(theStringDecay);

  theCascade = new G4GeneratorPrecompoundInterface;
  thePreEquilib = new G4PreCompoundModel(new G4ExcitationHandler);
  theCascade->SetDeExcitation(thePreEquilib);

  theModel->SetTransport(theCascade);
  theModel->SetHighEnergyGenerator(theStringModel);
  if (quasiElastic) {
    theQuasiElastic = new G4QuasiElasticChannel;
    theModel->SetQuasiElasticChannel(theQuasiElastic);
  } else {
    theQuasiElastic = nullptr;
  }
}

CMSQGSPSIMPBuilder::~CMSQGSPSIMPBuilder() {
  delete theStringDecay;
  delete theStringModel;
  delete thePreEquilib;
  delete theCascade;
  if (theQuasiElastic)
    delete theQuasiElastic;
  delete theModel;
  delete theQGSM;
}

void CMSQGSPSIMPBuilder::Build(G4HadronElasticProcess*) {}

void CMSQGSPSIMPBuilder::Build(G4HadronFissionProcess*) {}

void CMSQGSPSIMPBuilder::Build(G4HadronCaptureProcess*) {}

void CMSQGSPSIMPBuilder::Build(CMSSIMPInelasticProcess* aP) {
  theModel->SetMinEnergy(theMin);
  theModel->SetMaxEnergy(100 * TeV);
  aP->RegisterMe(theModel);
  aP->AddDataSet(new CMSSIMPInelasticXS());
}
