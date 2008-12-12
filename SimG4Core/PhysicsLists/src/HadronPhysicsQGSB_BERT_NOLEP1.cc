#include "SimG4Core/PhysicsLists/interface/HadronPhysicsQGSB_BERT_NOLEP1.hh"

#include "globals.hh"
#include "G4ios.hh"
#include <iomanip>   
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"

#include "G4MesonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4ShortLivedConstructor.hh"

HadronPhysicsQGSB_BERT_NOLEP1::HadronPhysicsQGSB_BERT_NOLEP1(const G4String& name, G4bool quasiElastic)
  :  G4VPhysicsConstructor(name) , QuasiElastic(quasiElastic) {
}

void HadronPhysicsQGSB_BERT_NOLEP1::CreateModels() {

  theNeutrons=new G4NeutronBuilder;
  theNeutrons->RegisterMe(theQGSBinaryNeutron=new G4QGSBinaryNeutronBuilder(QuasiElastic));
  theQGSBinaryNeutron->SetMinEnergy(8.5*GeV);
  theNeutrons->RegisterMe(theLEPNeutron=new G4LEPNeutronBuilder);
//   do not use LEP for inelastic, but leave capture, etc.
  theLEPNeutron->SetMinInelasticEnergy(0.*eV);
  theLEPNeutron->SetMaxInelasticEnergy(0.*eV);  
  theNeutrons->RegisterMe(theBertiniNeutron=new G4BertiniNeutronBuilder);
  theBertiniNeutron->SetMinEnergy(0.0*GeV);
  theBertiniNeutron->SetMaxEnergy(9.9*GeV);

  thePro=new G4ProtonBuilder;
  thePro->RegisterMe(theQGSBinaryPro=new G4QGSBinaryProtonBuilder(QuasiElastic));
  theQGSBinaryPro->SetMinEnergy(8.5*GeV);
  thePro->RegisterMe(theBertiniPro=new G4BertiniProtonBuilder);
  theBertiniPro->SetMaxEnergy(9.9*GeV);
  
  thePiK=new G4PiKBuilder;
  thePiK->RegisterMe(theQGSBinaryPiK=new G4QGSBinaryPiKBuilder(QuasiElastic));
  theQGSBinaryPiK->SetMinEnergy(8.5*GeV);

  thePiK->RegisterMe(theBertiniPiK=new G4BertiniPiKBuilder);
  theBertiniPiK->SetMaxEnergy(9.9*GeV);
  
  theMiscLHEP=new G4MiscLHEPBuilder;
}

HadronPhysicsQGSB_BERT_NOLEP1::~HadronPhysicsQGSB_BERT_NOLEP1() {
  delete theMiscLHEP;
  delete theQGSBinaryNeutron;
  delete theLEPNeutron;
  delete theBertiniNeutron;
  delete theQGSBinaryPro;
  delete thePro;
  delete theBertiniPro;
  delete theQGSBinaryPiK;
  delete theBertiniPiK;
  delete thePiK;
}

void HadronPhysicsQGSB_BERT_NOLEP1::ConstructParticle() {
  G4MesonConstructor pMesonConstructor;
  pMesonConstructor.ConstructParticle();

  G4BaryonConstructor pBaryonConstructor;
  pBaryonConstructor.ConstructParticle();

  G4ShortLivedConstructor pShortLivedConstructor;
  pShortLivedConstructor.ConstructParticle();  
}

#include "G4ProcessManager.hh"
void HadronPhysicsQGSB_BERT_NOLEP1::ConstructProcess() {
  CreateModels();
  theNeutrons->Build();
  thePro->Build();
  thePiK->Build();
  theMiscLHEP->Build();
}

