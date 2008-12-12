#ifndef HadronPhysicsQGSB_BERT_NOLEP1_h
#define HadronPhysicsQGSB_BERT_NOLEP1_h 1

#include "globals.hh"
#include "G4ios.hh"

#include "G4VPhysicsConstructor.hh"
#include "G4MiscLHEPBuilder.hh"

#include "G4PiKBuilder.hh"
#include "G4QGSBinaryPiKBuilder.hh"
#include "G4BertiniPiKBuilder.hh"

#include "G4ProtonBuilder.hh"
#include "G4QGSBinaryProtonBuilder.hh"
#include "G4BertiniProtonBuilder.hh"

#include "G4NeutronBuilder.hh"
#include "G4LEPNeutronBuilder.hh"
#include "G4QGSBinaryNeutronBuilder.hh"
#include "G4BertiniNeutronBuilder.hh"

class HadronPhysicsQGSB_BERT_NOLEP1 : public G4VPhysicsConstructor {

public: 
  HadronPhysicsQGSB_BERT_NOLEP1(const G4String& name ="hadron",G4bool quasiElastic=true);
  virtual ~HadronPhysicsQGSB_BERT_NOLEP1();

public: 
  virtual void ConstructParticle();
  virtual void ConstructProcess();

  void SetQuasiElastic(G4bool value) {QuasiElastic = value;}; 

private:
  void CreateModels();
  G4NeutronBuilder * theNeutrons;
  G4LEPNeutronBuilder * theLEPNeutron;
  G4QGSBinaryNeutronBuilder * theQGSBinaryNeutron;
  G4BertiniNeutronBuilder * theBertiniNeutron;
    
  G4PiKBuilder * thePiK;
  G4QGSBinaryPiKBuilder * theQGSBinaryPiK;
  G4BertiniPiKBuilder * theBertiniPiK;
  
  G4ProtonBuilder * thePro;
  G4QGSBinaryProtonBuilder * theQGSBinaryPro; 
  G4BertiniProtonBuilder * theBertiniPro;
    
  G4MiscLHEPBuilder * theMiscLHEP;
  
  G4bool QuasiElastic;
};

#endif

