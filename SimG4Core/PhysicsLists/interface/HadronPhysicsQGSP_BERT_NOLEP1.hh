#ifndef HadronPhysicsQGSP_BERT_NOLEP1_h
#define HadronPhysicsQGSP_BERT_NOLEP1_h 1

#include "globals.hh"
#include "G4ios.hh"

#include "G4VPhysicsConstructor.hh"
#include "G4MiscLHEPBuilder.hh"

#include "G4PiKBuilder.hh"
#include "G4QGSPPiKBuilder.hh"
#include "G4BertiniPiKBuilder.hh"

#include "G4ProtonBuilder.hh"
#include "G4QGSPProtonBuilder.hh"
#include "G4BertiniProtonBuilder.hh"

#include "G4NeutronBuilder.hh"
#include "G4LEPNeutronBuilder.hh"
#include "G4QGSPNeutronBuilder.hh"
#include "G4BertiniNeutronBuilder.hh"

class HadronPhysicsQGSP_BERT_NOLEP1 : public G4VPhysicsConstructor {

public: 
  HadronPhysicsQGSP_BERT_NOLEP1(const G4String& name ="hadron",G4bool quasiElastic=true);
  virtual ~HadronPhysicsQGSP_BERT_NOLEP1();

public: 
  virtual void ConstructParticle();
  virtual void ConstructProcess();

  void SetQuasiElastic(G4bool value) {QuasiElastic = value;}; 
  void SetProjectileDiffraction(G4bool value) {ProjectileDiffraction = value;};

private:
  void CreateModels();
  G4NeutronBuilder * theNeutrons;
  G4LEPNeutronBuilder * theLEPNeutron;
  G4QGSPNeutronBuilder * theQGSPNeutron;
  G4BertiniNeutronBuilder * theBertiniNeutron;
    
  G4PiKBuilder * thePiK;
  G4QGSPPiKBuilder * theQGSPPiK;
  G4BertiniPiKBuilder * theBertiniPiK;
  
  G4ProtonBuilder * thePro;
  G4QGSPProtonBuilder * theQGSPPro; 
  G4BertiniProtonBuilder * theBertiniPro;
    
  G4MiscLHEPBuilder * theMiscLHEP;
  
  G4bool QuasiElastic;
  G4bool ProjectileDiffraction;
};

#endif

