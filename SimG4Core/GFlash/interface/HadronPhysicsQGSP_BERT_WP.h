//
//---------------------------------------------------------------------------
//
// ClassName:   HadronPhysicsQGSP_BERT_WP
//
// Author: 2002 J.P. Wellisch
//
// Modified:
// 23.11.2005 G.Folger: migration to non static particles
// 08.06.2006 V.Ivanchenko: remove stopping
// 25.04.2007 G.Folger: Add quasielastic option, use this by default
// 10.12.2007 G.Folger: Add projectilediffrative option for proton/neutron, off by default
//
//----------------------------------------------------------------------------
//
#ifndef HadronPhysicsQGSP_BERT_WP_h
#define HadronPhysicsQGSP_BERT_WP_h 1

#include "globals.hh"
#include "G4ios.hh"

#include "G4VPhysicsConstructor.hh"
//#include "G4MiscLHEPBuilder.hh"

//#include "G4LEPPiKBuilder.hh"
#include "G4QGSPPiKBuilder.hh"
#include "G4BertiniPiKBuilder.hh"

#include "G4ProtonBuilder.hh"
//#include "G4LEPProtonBuilder.hh"
#include "G4QGSPProtonBuilder.hh"
#include "G4BertiniProtonBuilder.hh"

#include "G4NeutronBuilder.hh"
//#include "G4LEPNeutronBuilder.hh"
#include "G4QGSPNeutronBuilder.hh"
#include "G4BertiniNeutronBuilder.hh"

class G4PiKBuilder_WP;
class G4ProtonBuilder_WP;
class G4MiscLHEPBuilder_WP;

class HadronPhysicsQGSP_BERT_WP : public G4VPhysicsConstructor
{
  public: 
    HadronPhysicsQGSP_BERT_WP(const G4String& name ="hadron",G4bool quasiElastic=true);
    virtual ~HadronPhysicsQGSP_BERT_WP();

  public: 
    virtual void ConstructParticle();
    virtual void ConstructProcess();

    void SetQuasiElastic(G4bool value) {QuasiElastic = value;}; 
    void SetProjectileDiffraction(G4bool value) {ProjectileDiffraction = value;}; 

  private:
    void CreateModels();
    G4NeutronBuilder * theNeutrons;
    //G4LEPNeutronBuilder * theLEPNeutron;
    G4QGSPNeutronBuilder * theQGSPNeutron;
    G4BertiniNeutronBuilder * theBertiniNeutron;
    
    G4PiKBuilder_WP * thePiK;
    // G4LEPPiKBuilder * theLEPPiK;
    G4QGSPPiKBuilder * theQGSPPiK;
    G4BertiniPiKBuilder * theBertiniPiK;
    
    G4ProtonBuilder_WP * thePro;
    // G4LEPProtonBuilder * theLEPPro;
    G4QGSPProtonBuilder * theQGSPPro; 
    G4BertiniProtonBuilder * theBertiniPro;
    
    G4MiscLHEPBuilder_WP * theMiscLHEP;
    
    G4bool QuasiElastic;
    G4bool ProjectileDiffraction;
};
// 2008 Modified for CMS GflashHadronWrapperProcess
#endif

