#include "SimG4Core/PhysicsLists/interface/HadronPhysicsQGSPCMS_FTFP_BERT.h"
#include "G4SystemOfUnits.hh"
#include "G4Threading.hh"

HadronPhysicsQGSPCMS_FTFP_BERT::HadronPhysicsQGSPCMS_FTFP_BERT(G4int)
    : HadronPhysicsQGSPCMS_FTFP_BERT(
          3. * CLHEP::GeV, 6. * CLHEP::GeV, 12. * CLHEP::GeV, 25. * CLHEP::GeV, 12. * CLHEP::GeV) {}

HadronPhysicsQGSPCMS_FTFP_BERT::HadronPhysicsQGSPCMS_FTFP_BERT(
    G4double e1, G4double e2, G4double e3, G4double e4, G4double e5)
    : G4HadronPhysicsQGSP_BERT("hInelasticQGSPCMS_FTFP_BERT") {
  minQGSP_proton = minQGSP_neutron = minQGSP_pik = e5;
  maxFTFP_proton = maxFTFP_neutron = maxFTFP_pik = e4;
  minFTFP_proton = minFTFP_neutron = minFTFP_pik = e1;
  maxBERT_proton = maxBERT_neutron = e2;
  maxBERT_pik = e3;
}

HadronPhysicsQGSPCMS_FTFP_BERT::~HadronPhysicsQGSPCMS_FTFP_BERT() {}

void HadronPhysicsQGSPCMS_FTFP_BERT::ConstructProcess() {
  if (G4Threading::IsMasterThread()) {
    DumpBanner();
  }
  CreateModels();
}
