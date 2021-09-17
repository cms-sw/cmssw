//---------------------------------------------------------------------------
// Author: Vladimir Ivanchenko
// Date:   March 2018
//
// Hadron physics for the new CMS physics list FTFP_BERT_EMM_TRK.
// The hadron physics of FTFP_BERT has the transition between Bertini
// (BERT) intra-nuclear cascade model and Fritiof (FTF) string model in the
// energy region [4, 5] GeV (instead of the default for Geant4 10.4).
//---------------------------------------------------------------------------
//
#ifndef SimG4Core_PhysicsLists_CMSHadronPhysicsFTFP_BERT106_h
#define SimG4Core_PhysicsLists_CMSHadronPhysicsFTFP_BERT106_h

#include "globals.hh"
#include "G4ios.hh"

#include "G4VPhysicsConstructor.hh"

class CMSHadronPhysicsFTFP_BERT106 : public G4VPhysicsConstructor {
public:
  explicit CMSHadronPhysicsFTFP_BERT106(G4int verb);
  explicit CMSHadronPhysicsFTFP_BERT106(G4double e1, G4double e2, G4double e3);
  ~CMSHadronPhysicsFTFP_BERT106() override;

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
  G4double maxBERTpi_;
};

#endif
