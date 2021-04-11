//---------------------------------------------------------------------------
// Author: Vladimir Ivanchenko
// Date:   March 2018
//
// Hadron physics for the new CMS physics list FTFP_BERT_EMM_TRK.
// The hadron physics of FTFP_BERT has the transition between Bertini
// (BERT) intra-nuclear cascade model and Fritiof (FTF) string model
// optimized for CMS.
//---------------------------------------------------------------------------
//
#ifndef SimG4Core_PhysicsLists_CMSHadronPhysicsFTFP_BERT_h
#define SimG4Core_PhysicsLists_CMSHadronPhysicsFTFP_BERT_h

#include "globals.hh"
#include "G4ios.hh"

#include "G4HadronPhysicsFTFP_BERT.hh"

class CMSHadronPhysicsFTFP_BERT : public G4HadronPhysicsFTFP_BERT {
public:
  explicit CMSHadronPhysicsFTFP_BERT(G4int verb);
  explicit CMSHadronPhysicsFTFP_BERT(G4double e1, G4double e2, G4double e3);
  ~CMSHadronPhysicsFTFP_BERT() override;

  void ConstructProcess() override;

  // copy constructor and hide assignment operator
  CMSHadronPhysicsFTFP_BERT(CMSHadronPhysicsFTFP_BERT &) = delete;
  CMSHadronPhysicsFTFP_BERT &operator=(const CMSHadronPhysicsFTFP_BERT &right) = delete;
};

#endif
