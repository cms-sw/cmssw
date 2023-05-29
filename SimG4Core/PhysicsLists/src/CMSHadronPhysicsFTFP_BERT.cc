
#include "SimG4Core/PhysicsLists/interface/CMSHadronPhysicsFTFP_BERT.h"

#include "G4TheoFSGenerator.hh"
#include "G4FTFModel.hh"
#include "G4ExcitedStringDecay.hh"
#include "G4GeneratorPrecompoundInterface.hh"
#include "G4CascadeInterface.hh"

#include "G4HadronicParameters.hh"
#include "G4HadronicProcess.hh"
#include "G4HadronInelasticProcess.hh"
#include "G4HadProcesses.hh"
#include "G4SystemOfUnits.hh"
#include "G4Threading.hh"

CMSHadronPhysicsFTFP_BERT::CMSHadronPhysicsFTFP_BERT(G4int)
    : CMSHadronPhysicsFTFP_BERT(3 * CLHEP::GeV, 6 * CLHEP::GeV, 12 * CLHEP::GeV, 3 * CLHEP::GeV, 6 * CLHEP::GeV) {}

CMSHadronPhysicsFTFP_BERT::CMSHadronPhysicsFTFP_BERT(G4double e1, G4double e2, G4double e3, G4double e4, G4double e5)
    : G4HadronPhysicsFTFP_BERT("hInelastic FTFP_BERT", false) {
  minFTFP_pion = e1;
  maxBERT_pion = e3;
  minFTFP_kaon = e1;
  maxBERT_kaon = e2;
  minFTFP_kaon = e4;
  maxBERT_kaon = e5;
  minFTFP_proton = e1;
  maxBERT_proton = e2;
  minFTFP_neutron = e1;
  maxBERT_neutron = e2;
}

void CMSHadronPhysicsFTFP_BERT::ConstructProcess() {
  if (G4Threading::IsMasterThread()) {
    DumpBanner();
  }
  CreateModels();
}

void CMSHadronPhysicsFTFP_BERT::Neutron() {
  G4bool useNGeneral = G4HadronicParameters::Instance()->EnableNeutronGeneralProcess();
  if (useNGeneral) {
    auto theFTFP = new G4TheoFSGenerator("FTFP");
    auto theStringModel = new G4FTFModel();
    theStringModel->SetFragmentationModel(new G4ExcitedStringDecay());
    theFTFP->SetHighEnergyGenerator(theStringModel);
    theFTFP->SetTransport(new G4GeneratorPrecompoundInterface());
    theFTFP->SetMinEnergy(minFTFP_neutron);
    theFTFP->SetMaxEnergy(G4HadronicParameters::Instance()->GetMaxEnergy());

    auto theBERT = new G4CascadeInterface();
    theBERT->SetMaxEnergy(maxBERT_neutron);

    G4HadronicProcess* ni = new G4HadronInelasticProcess("neutronInelastic", G4Neutron::Neutron());
    ni->RegisterMe(theFTFP);
    ni->RegisterMe(theBERT);
    G4HadProcesses::BuildNeutronInelasticAndCapture(ni);
    return;
  }

  G4HadronPhysicsFTFP_BERT::Neutron();
}
