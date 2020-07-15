#ifndef SimG4Core_PhysicsLists_HadronPhysicsQGSPCMS_FTFP_BERT_h
#define SimG4Core_PhysicsLists_HadronPhysicsQGSPCMS_FTFP_BERT_h 1

#include "globals.hh"
#include "G4ios.hh"

#include "G4VPhysicsConstructor.hh"

class HadronPhysicsQGSPCMS_FTFP_BERT : public G4VPhysicsConstructor {
public:
  explicit HadronPhysicsQGSPCMS_FTFP_BERT(G4int verbose);
  explicit HadronPhysicsQGSPCMS_FTFP_BERT(G4double e1, G4double e2, G4double e3, G4double e4, G4double e5);
  ~HadronPhysicsQGSPCMS_FTFP_BERT() override;

  void ConstructParticle() override;
  void ConstructProcess() override;

private:
  //This calls the specific ones for the different particles in order
  void CreateModels();
  void Neutron();
  void Proton();
  void Pion();
  void Kaon();
  void Others();
  void DumpBanner();
  //This contains extra configurataion specific to this PL
  void ExtraConfiguration();

  G4double minFTFP_;
  G4double maxBERT_;
  G4double minQGSP_;
  G4double maxFTFP_;
  G4double maxBERTpi_;
};

#endif
