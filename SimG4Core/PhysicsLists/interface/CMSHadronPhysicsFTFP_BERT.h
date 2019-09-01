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
#ifndef SimG4Core_PhysicsLists_CMSHadronPhysicsFTFP_BERT_h
#define SimG4Core_PhysicsLists_CMSHadronPhysicsFTFP_BERT_h

#include "globals.hh"
#include "G4ios.hh"

#include "G4VPhysicsConstructor.hh"

#include "G4Cache.hh"

class G4ComponentGGHadronNucleusXsc;
class G4VCrossSectionDataSet;

class CMSHadronPhysicsFTFP_BERT : public G4VPhysicsConstructor {
public:
  explicit CMSHadronPhysicsFTFP_BERT(G4int verbose = 1);
  ~CMSHadronPhysicsFTFP_BERT() override;

  void ConstructParticle() override;
  //This will call in order:
  // DumpBanner (for master)
  // CreateModels
  // ExtraConfiguation
  void ConstructProcess() override;

  void TerminateWorker() override;

protected:
  G4bool QuasiElastic;
  //This calls the specific ones for the different particles in order
  virtual void CreateModels();
  virtual void Neutron();
  virtual void Proton();
  virtual void Pion();
  virtual void Kaon();
  virtual void Others();
  virtual void DumpBanner();
  //This contains extra configurataion specific to this PL
  virtual void ExtraConfiguration();

  G4double minFTFP_pion;
  G4double maxBERT_pion;
  G4double minFTFP_kaon;
  G4double maxBERT_kaon;
  G4double minFTFP_proton;
  G4double maxBERT_proton;
  G4double minFTFP_neutron;
  G4double maxBERT_neutron;

  //Thread-private data write them here to delete them
  G4VectorCache<G4VCrossSectionDataSet*> xs_ds;
  G4Cache<G4ComponentGGHadronNucleusXsc*> xs_k;
};

#endif
