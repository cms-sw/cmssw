
#include "SimG4Core/PhysicsLists/interface/CMSHadronPhysicsFTFP_BERT.h"
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

CMSHadronPhysicsFTFP_BERT::~CMSHadronPhysicsFTFP_BERT() {}

void CMSHadronPhysicsFTFP_BERT::ConstructProcess() {
  if (G4Threading::IsMasterThread()) {
    DumpBanner();
  }
  CreateModels();
}
