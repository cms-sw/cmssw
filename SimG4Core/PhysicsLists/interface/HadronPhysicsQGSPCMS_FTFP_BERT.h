#ifndef SimG4Core_PhysicsLists_HadronPhysicsQGSPCMS_FTFP_BERT_h
#define SimG4Core_PhysicsLists_HadronPhysicsQGSPCMS_FTFP_BERT_h 1

#include "globals.hh"
#include "G4ios.hh"

#include "G4VPhysicsConstructor.hh"

#include "G4PiKBuilder.hh"
#include "G4FTFPPiKBuilder.hh"
#include "G4QGSPPiKBuilder.hh"
#include "G4BertiniPiKBuilder.hh"

#include "G4ProtonBuilder.hh"
#include "G4FTFPProtonBuilder.hh"
#include "G4QGSPProtonBuilder.hh"
#include "G4BertiniProtonBuilder.hh"

#include "G4NeutronBuilder.hh"
#include "G4FTFPNeutronBuilder.hh"
#include "G4QGSPNeutronBuilder.hh"
#include "G4BertiniNeutronBuilder.hh"

#include "G4HyperonFTFPBuilder.hh"
#include "G4AntiBarionBuilder.hh"
#include "G4FTFPAntiBarionBuilder.hh"

class HadronPhysicsQGSPCMS_FTFP_BERT : public G4VPhysicsConstructor
{
  public:
    HadronPhysicsQGSPCMS_FTFP_BERT(G4int verbose);
    virtual ~HadronPhysicsQGSPCMS_FTFP_BERT();

    virtual void ConstructParticle();
    virtual void ConstructProcess();

  private:
    void CreateModels();

    struct ThreadPrivate {
      G4NeutronBuilder * theNeutrons;
      G4FTFPNeutronBuilder * theFTFPNeutron;
      G4QGSPNeutronBuilder * theQGSPNeutron;
      G4BertiniNeutronBuilder * theBertiniNeutron;
    
      G4PiKBuilder * thePiK;
      G4FTFPPiKBuilder * theFTFPPiK;
      G4QGSPPiKBuilder * theQGSPPiK;
      G4BertiniPiKBuilder * theBertiniPiK;
    
      G4ProtonBuilder * thePro;
      G4FTFPProtonBuilder * theFTFPPro;
      G4QGSPProtonBuilder * theQGSPPro; 
      G4BertiniProtonBuilder * theBertiniPro;
    
      G4HyperonFTFPBuilder *theHyperon;

      G4AntiBarionBuilder     *theAntiBaryon;
      G4FTFPAntiBarionBuilder *theFTFPAntiBaryon;
    };
    static G4ThreadLocal ThreadPrivate* tpdata;    
};

#endif

