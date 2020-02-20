//
#ifndef CMSHadronPhysicsFTFP_BERT_h
#define CMSHadronPhysicsFTFP_BERT_h 1

#include "globals.hh"
#include "G4ios.hh"

#include "G4VPhysicsConstructor.hh"

class CMSHadronPhysicsFTFP_BERT : public G4VPhysicsConstructor {
public:
  explicit CMSHadronPhysicsFTFP_BERT(G4int verb);
  explicit CMSHadronPhysicsFTFP_BERT(G4double e1, G4double e2);
  ~CMSHadronPhysicsFTFP_BERT() override;

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
};

#endif
