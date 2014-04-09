#ifndef SimG4Core_PhysicsLists_HadronPhysicsQGSPCMS_FTFP_BERT_WP_h
#define SimG4Core_PhysicsLists_HadronPhysicsQGSPCMS_FTFP_BERT_WP_h 1

#include "globals.hh"
#include "G4ios.hh"

#include "G4VPhysicsConstructor.hh"
//#include "G4MiscLHEPBuilder.hh"

#include "G4PiKBuilder.hh"
#include "SimG4Core/PhysicsLists/interface/CMSFTFPPiKBuilder.hh"
#include "G4QGSPPiKBuilder.hh"
#include "G4BertiniPiKBuilder.hh"

#include "G4ProtonBuilder.hh"
#include "SimG4Core/PhysicsLists/interface/CMSFTFPProtonBuilder.hh"
#include "G4QGSPProtonBuilder.hh"
#include "G4BertiniProtonBuilder.hh"

#include "G4NeutronBuilder.hh"
#include "SimG4Core/PhysicsLists/interface/CMSFTFPNeutronBuilder.hh"
#include "G4QGSPNeutronBuilder.hh"
#include "G4BertiniNeutronBuilder.hh"
//#include "G4LEPNeutronBuilder.hh"

class G4PiKBuilder_WP;
class G4ProtonBuilder_WP;
class G4MiscLHEPBuilder_WP;

class HadronPhysicsQGSPCMS_FTFP_BERT_WP : public G4VPhysicsConstructor
{
  public: 
    HadronPhysicsQGSPCMS_FTFP_BERT_WP(const G4String& name ="hadron",G4bool quasiElastic=true);
    virtual ~HadronPhysicsQGSPCMS_FTFP_BERT_WP();

  public: 
    virtual void ConstructParticle();
    virtual void ConstructProcess();

    void SetQuasiElastic(G4bool value) {QuasiElastic = value;}; 
    void SetProjectileDiffraction(G4bool value) {ProjectileDiffraction = value;}; 

  private:
    void CreateModels();
    G4NeutronBuilder * theNeutrons;
    CMSFTFPNeutronBuilder * theFTFPNeutron;
    G4QGSPNeutronBuilder * theQGSPNeutron;
    G4BertiniNeutronBuilder * theBertiniNeutron;
    //G4LEPNeutronBuilder * theLEPNeutron;
    
    G4PiKBuilder_WP * thePiK;
    CMSFTFPPiKBuilder * theFTFPPiK;
    G4QGSPPiKBuilder * theQGSPPiK;
    G4BertiniPiKBuilder * theBertiniPiK;
    
    G4ProtonBuilder_WP * thePro;
    CMSFTFPProtonBuilder * theFTFPPro;
    G4QGSPProtonBuilder * theQGSPPro; 
    G4BertiniProtonBuilder * theBertiniPro;
    
    G4MiscLHEPBuilder_WP * theMiscLHEP;
    
    G4bool QuasiElastic;
    G4bool ProjectileDiffraction;
};
// 2011 Mar 3 Modified for CMS GflashHadronWrapperProcess
#endif

