#include "SimG4Core/PhysicsLists/interface/HadronPhysicsCMS.h"

#include "globals.hh"
#include "G4ios.hh"
#include <iomanip>   
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"

#include "G4MesonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4ShortLivedConstructor.hh"
#include "G4SystemOfUnits.hh"

HadronPhysicsCMS::HadronPhysicsCMS(const G4String& name, G4bool quasiElastic) :
  G4VPhysicsConstructor("hadron"), theNeutrons(0), theBertiniNeutron(0),
  theBinaryNeutron(0), theFTFPNeutron(0), 
  theQGSPNeutron(0), thePiK(0), theBertiniPiK(0),
  theBinaryPiK(0), theFTFPPiK(0), 
  theQGSPPiK(0), thePro(0),theBertiniPro(0),
  theBinaryPro(0), theFTFPPro(0), 
  theQGSPPro(0),
  theFTFNeutron(0), theFTFPiK(0), theFTFPro(0), 
  modelName(name), 
  QuasiElastic(quasiElastic) {}

void HadronPhysicsCMS::CreateModels() {

  theNeutrons = new G4NeutronBuilder;
  thePro      = new G4ProtonBuilder;
  thePiK      = new G4PiKBuilder;

  if (modelName == "Bertini") {
    theBertiniNeutron = new G4BertiniNeutronBuilder();
    theBertiniNeutron->SetMaxEnergy(30.0*GeV);
    theNeutrons->RegisterMe(theBertiniNeutron);
    theBertiniPro     = new G4BertiniProtonBuilder();
    theBertiniPro->SetMaxEnergy(30.0*GeV);
    thePro->RegisterMe(theBertiniPro);
    theBertiniPiK     = new G4BertiniPiKBuilder();
    theBertiniPiK->SetMaxEnergy(30.0*GeV);
    thePiK->RegisterMe(theBertiniPiK);
  } else if (modelName == "Binary") {
    theBinaryNeutron = new G4BinaryNeutronBuilder();
    theBinaryNeutron->SetMaxEnergy(30.0*GeV);
    theNeutrons->RegisterMe(theBinaryNeutron);
    theBinaryPro     = new G4BinaryProtonBuilder();
    theBinaryPro->SetMaxEnergy(30.0*GeV);
    thePro->RegisterMe(theBinaryPro);
    theBinaryPiK     = new G4BinaryPiKBuilder(); 
    theBinaryPiK->SetMaxEnergy(30.0*GeV);
    thePiK->RegisterMe(theBinaryPiK);
  } else if (modelName == "FTFP") {
    theFTFPNeutron = new G4FTFPNeutronBuilder();
    theFTFPNeutron->SetMinEnergy(0.1*GeV);
    theNeutrons->RegisterMe(theFTFPNeutron);
    theFTFPPro     = new G4FTFPProtonBuilder();
    theFTFPPro->SetMinEnergy(0.1*GeV);
    thePro->RegisterMe(theFTFPPro);
    theFTFPPiK     = new G4FTFPPiKBuilder();
    theFTFPPiK->SetMinEnergy(0.1*GeV);
    thePiK->RegisterMe(theFTFPPiK);
  } else if (modelName == "FTF") {
    theFTFNeutron  = new G4FTFBinaryNeutronBuilder();
    theNeutrons->RegisterMe(theFTFNeutron);
    theFTFPro      = new G4FTFBinaryProtonBuilder();
    thePro->RegisterMe(theFTFPro);
    theFTFPiK      = new G4FTFBinaryPiKBuilder();
    thePiK->RegisterMe(theFTFPiK);
  } else {
    theQGSPNeutron = new G4QGSPNeutronBuilder(QuasiElastic);
    theQGSPNeutron->SetMinEnergy(0.1*GeV);
    theNeutrons->RegisterMe(theQGSPNeutron);
    theQGSPPro     = new G4QGSPProtonBuilder(QuasiElastic);
    theQGSPPro->SetMinEnergy(0.1*GeV);
    thePro->RegisterMe(theQGSPPro);
    theQGSPPiK     = new G4QGSPPiKBuilder(QuasiElastic);
    theQGSPPiK->SetMinEnergy(0.1*GeV);
    thePiK->RegisterMe(theQGSPPiK);
  }
  
}

HadronPhysicsCMS::~HadronPhysicsCMS() {
  if (theBertiniNeutron)   delete theBertiniNeutron;
  if (theBinaryNeutron)    delete theBinaryNeutron;
  if (theFTFPNeutron)      delete theFTFPNeutron;
  if (theQGSPNeutron)      delete theQGSPNeutron;
  if (theFTFNeutron)       delete theFTFNeutron;
  delete theNeutrons;
  if (theBertiniPro)       delete theBertiniPro;
  if (theBinaryPro)        delete theBinaryPro;
  if (theFTFPPro)          delete theFTFPPro;
  if (theQGSPPro)          delete theQGSPPro; 
  if (theFTFPro)           delete theFTFPro;
  delete thePro;
  if (theBertiniPiK)       delete theBertiniPiK;
  if (theBinaryPiK)        delete theBinaryPiK;
  if (theFTFPPiK)          delete theFTFPPiK;
  if (theQGSPPiK)          delete theQGSPPiK;
  if (theFTFPiK)           delete theFTFPiK;
  delete thePiK;
}


void HadronPhysicsCMS::ConstructParticle() {

  G4MesonConstructor pMesonConstructor;
  pMesonConstructor.ConstructParticle();

  G4BaryonConstructor pBaryonConstructor;
  pBaryonConstructor.ConstructParticle();

  G4ShortLivedConstructor pShortLivedConstructor;
  pShortLivedConstructor.ConstructParticle();  
}

#include "G4ProcessManager.hh"
void HadronPhysicsCMS::ConstructProcess() {

  CreateModels();
  theNeutrons->Build();
  thePro->Build();
  thePiK->Build();
  //  theMiscLHEP->Build();
}

