//--------------------------------------------------------------------
//
// 15.04.2021 V.Ivanchenko Hadron inelastic physics based on
//                         QGSP_FTFP_BERT of CMS migrated to Geant4 10.7
//
//--------------------------------------------------------------------

#ifndef SimG4Core_PhysicsLists_HadronPhysicsQGSPCMS_FTFP_BERT_h
#define SimG4Core_PhysicsLists_HadronPhysicsQGSPCMS_FTFP_BERT_h 1

#include "globals.hh"
#include "G4ios.hh"

#include "G4HadronPhysicsQGSP_BERT.hh"

class HadronPhysicsQGSPCMS_FTFP_BERT : public G4HadronPhysicsQGSP_BERT {
public:
  explicit HadronPhysicsQGSPCMS_FTFP_BERT(G4int verbose);
  explicit HadronPhysicsQGSPCMS_FTFP_BERT(G4double e1, G4double e2, G4double e3, G4double e4, G4double e5);
  ~HadronPhysicsQGSPCMS_FTFP_BERT() override;

  void ConstructProcess() override;

  // copy constructor and hide assignment operator
  HadronPhysicsQGSPCMS_FTFP_BERT(HadronPhysicsQGSPCMS_FTFP_BERT &) = delete;
  HadronPhysicsQGSPCMS_FTFP_BERT &operator=(const HadronPhysicsQGSPCMS_FTFP_BERT &right) = delete;
};

#endif
