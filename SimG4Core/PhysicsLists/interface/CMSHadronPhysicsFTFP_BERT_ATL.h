//---------------------------------------------------------------------------
// Author: Alberto Ribon
// Date:   April 2016
//
// Hadron physics for the new physics list FTFP_BERT_ATL.
// This is a modified version of the FTFP_BERT hadron physics for ATLAS.
// The hadron physics of FTFP_BERT_ATL has the transition between Bertini
// (BERT) intra-nuclear cascade model and Fritiof (FTF) string model in the
// energy region [9, 12] GeV (instead of [4, 5] GeV as in FTFP_BERT).
//---------------------------------------------------------------------------
//
#ifndef SimG4Core_PhysicsLists_CMSHadronPhysicsFTFP_BERT_ATL_h
#define SimG4Core_PhysicsLists_CMSHadronPhysicsFTFP_BERT_ATL_h

#include "globals.hh"
#include "G4ios.hh"

#include "G4VPhysicsConstructor.hh"

#include "G4PiKBuilder.hh"
#include "G4BertiniPiKBuilder.hh"
#include "G4FTFPPiKBuilder.hh"

#include "G4ProtonBuilder.hh"
#include "G4BertiniProtonBuilder.hh"
#include "G4FTFPNeutronBuilder.hh"
#include "G4FTFPProtonBuilder.hh"

#include "G4NeutronBuilder.hh"
#include "G4BertiniNeutronBuilder.hh"
#include "G4FTFPNeutronBuilder.hh"

#include "G4HyperonFTFPBuilder.hh"
#include "G4AntiBarionBuilder.hh"
#include "G4FTFPAntiBarionBuilder.hh"

class G4ComponentGGHadronNucleusXsc;

class CMSHadronPhysicsFTFP_BERT_ATL : public G4VPhysicsConstructor
{
  public:
    CMSHadronPhysicsFTFP_BERT_ATL(G4int verbose =1);
    ~CMSHadronPhysicsFTFP_BERT_ATL() override;

    void ConstructParticle() override;
    void ConstructProcess() override;

  private:
    void CreateModels();
    G4bool QuasiElastic;

    // Simplify handling of TLS data, encapsulate everyhing in a structure
    struct ThreadPrivate { 
      G4NeutronBuilder * theNeutrons;
      G4BertiniNeutronBuilder * theBertiniNeutron;
      G4FTFPNeutronBuilder * theFTFPNeutron;
 
      G4PiKBuilder * thePiK;
      G4BertiniPiKBuilder * theBertiniPiK;
      G4FTFPPiKBuilder * theFTFPPiK;
    
      G4ProtonBuilder * thePro;
      G4BertiniProtonBuilder * theBertiniPro;
      G4FTFPProtonBuilder * theFTFPPro;    
    
      G4HyperonFTFPBuilder * theHyperon;
    
      G4AntiBarionBuilder * theAntiBaryon;
      G4FTFPAntiBarionBuilder * theFTFPAntiBaryon;

      G4ComponentGGHadronNucleusXsc * xsKaon;
      G4VCrossSectionDataSet * xsNeutronInelasticXS;
      G4VCrossSectionDataSet * xsNeutronCaptureXS;
    };
    static G4ThreadLocal ThreadPrivate* tpdata;
};

#endif

