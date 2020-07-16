//---------------------------------------------------------------------------
// Author: Vladimir Ivanchenko
// Date:   June 2020
//
// Hyperon physics for the new CMS physics list FTFP_BERT_EMM
// The hadron physics of FTFP_BERT has the transition between Bertini
// (BERT) intra-nuclear cascade model and Fritiof (FTF) string model in the
// energy region defined by hadronic parametes
//---------------------------------------------------------------------------

#ifndef SimG4Core_PhysicsLists_CMSHyperonFTFPBuilder_h
#define SimG4Core_PhysicsLists_CMSHyperonFTFPBuilder_h

#include "G4PhysicsBuilderInterface.hh"
#include "globals.hh"

class CMSHyperonFTFPBuilder : public G4PhysicsBuilderInterface {
public:
  CMSHyperonFTFPBuilder();
  ~CMSHyperonFTFPBuilder() override;

  void Build() final;
};

#endif
