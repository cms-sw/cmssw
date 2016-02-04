#include "SimG4Core/PhysicsLists/interface/HadronPhysicsCMS.h"

#include "globals.hh"
#include "G4ios.hh"
#include <iomanip>   
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"

#include "G4MesonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4ShortLivedConstructor.hh"

HadronPhysicsCMS::HadronPhysicsCMS(const G4String& name, G4bool quasiElastic) :
  G4VPhysicsConstructor("hadron"), theNeutrons(0), theBertiniNeutron(0),
  theBinaryNeutron(0), theFTFCNeutron(0), theFTFPNeutron(0), theLEPNeutron(0),
  theLHEPNeutron(0), thePrecoNeutron(0), theQGSCEflowNeutron(0),
  theQGSCNeutron(0), theQGSPNeutron(0), thePiK(0), theBertiniPiK(0),
  theBinaryPiK(0), theFTFCPiK(0), theFTFPPiK(0), theLEPPiK(0), theLHEPPiK(0),
  theQGSCEflowPiK(0), theQGSCPiK(0), theQGSPPiK(0), thePro(0),theBertiniPro(0),
  theBinaryPro(0), theFTFCPro(0), theFTFPPro(0), theLEPPro(0), theLHEPPro(0),
  thePrecoPro(0),  theQGSCEflowPro(0), theQGSCPro(0), theQGSPPro(0),
  theMiscLHEP(), theFTFNeutron(0), theFTFPiK(0), theFTFPro(0), 
  theRPGNeutron(0), theRPGPiK(0), theRPGPro(0), modelName(name), 
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
  } else if (modelName == "FTFC") {
    theFTFCNeutron = new G4FTFCNeutronBuilder();
    theFTFCNeutron->SetMinEnergy(0.1*GeV);
    theNeutrons->RegisterMe(theFTFCNeutron);
    theFTFCPro     = new G4FTFCProtonBuilder();
    theFTFCPro->SetMinEnergy(0.1*GeV);
    thePro->RegisterMe(theFTFCPro);
    theFTFCPiK     = new G4FTFCPiKBuilder();
    theFTFCPiK->SetMinEnergy(0.1*GeV);
    thePiK->RegisterMe(theFTFCPiK);
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
  }  else if (modelName == "LEP") {
    theLEPNeutron = new G4LEPNeutronBuilder();
    theNeutrons->RegisterMe(theLEPNeutron);
    theLEPPro     = new G4LEPProtonBuilder();
    thePro->RegisterMe(theLEPPro);
    theLEPPiK     = new G4LEPPiKBuilder();
    thePiK->RegisterMe(theLEPPiK);
  }  else if (modelName == "LHEP") {
    theLHEPNeutron = new G4LHEPNeutronBuilder();
    theNeutrons->RegisterMe(theLHEPNeutron);
    theLHEPPro     = new G4LHEPProtonBuilder();
    thePro->RegisterMe(theLHEPPro);
    theLHEPPiK     = new G4LHEPPiKBuilder();
    thePiK->RegisterMe(theLHEPPiK);
  }  else if (modelName == "Preco") {
    thePrecoNeutron = new G4PrecoNeutronBuilder();
    theNeutrons->RegisterMe(thePrecoNeutron);
    thePrecoPro     = new G4PrecoProtonBuilder();
    thePro->RegisterMe(thePrecoPro);
    theLHEPPiK      = new G4LHEPPiKBuilder();
    thePiK->RegisterMe(theLHEPPiK);
  }  else if (modelName == "QGSCEflow") {
    theQGSCEflowNeutron = new G4QGSCEflowNeutronBuilder();
    theQGSCEflowNeutron->SetMinEnergy(0.1*GeV);
    theNeutrons->RegisterMe(theQGSCEflowNeutron);
    theQGSCEflowPro     = new G4QGSCEflowProtonBuilder();
    theQGSCEflowPro->SetMinEnergy(0.1*GeV);
    thePro->RegisterMe(theQGSCEflowPro);
    theQGSCEflowPiK     = new G4QGSCEflowPiKBuilder();
    theQGSCEflowPiK->SetMinEnergy(0.1*GeV);
    thePiK->RegisterMe(theQGSCEflowPiK);
  }  else if (modelName == "QGSC") {
    theQGSCNeutron = new G4QGSCNeutronBuilder();
    theQGSCNeutron->SetMinEnergy(0.1*GeV);
    theNeutrons->RegisterMe(theQGSCNeutron);
    theQGSCPro     = new G4QGSCProtonBuilder();
    theQGSCPro->SetMinEnergy(0.1*GeV);
    thePro->RegisterMe(theQGSCPro);
    theQGSCPiK     = new G4QGSCPiKBuilder();
    theQGSCPiK->SetMinEnergy(0.1*GeV);
    thePiK->RegisterMe(theQGSCPiK);
  } else if (modelName == "RPG") {
    theRPGNeutron  = new G4RPGNeutronBuilder();
    theNeutrons->RegisterMe(theRPGNeutron);
    theRPGPro      = new G4RPGProtonBuilder();
    thePro->RegisterMe(theRPGPro);
    theRPGPiK      = new G4RPGPiKBuilder();
    thePiK->RegisterMe(theRPGPiK);
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
  
  theMiscLHEP=new G4MiscLHEPBuilder;
}

HadronPhysicsCMS::~HadronPhysicsCMS() {
  delete theMiscLHEP;
  if (theBertiniNeutron)   delete theBertiniNeutron;
  if (theBinaryNeutron)    delete theBinaryNeutron;
  if (theFTFCNeutron)      delete theFTFCNeutron;
  if (theFTFPNeutron)      delete theFTFPNeutron;
  if (theLEPNeutron)       delete theLEPNeutron;
  if (theLHEPNeutron)      delete theLHEPNeutron;
  if (thePrecoNeutron)     delete thePrecoNeutron;
  if (theQGSCEflowNeutron) delete theQGSCEflowNeutron;
  if (theQGSCNeutron)      delete theQGSCNeutron;
  if (theQGSPNeutron)      delete theQGSPNeutron;
  if (theFTFNeutron)       delete theFTFNeutron;
  if (theRPGNeutron)       delete theRPGNeutron;
  delete theNeutrons;
  if (theBertiniPro)       delete theBertiniPro;
  if (theBinaryPro)        delete theBinaryPro;
  if (theFTFCPro)          delete theFTFCPro;
  if (theFTFPPro)          delete theFTFPPro;
  if (theLEPPro)           delete theLEPPro;
  if (theLHEPPro)          delete theLHEPPro;
  if (thePrecoPro)         delete thePrecoPro;
  if (theQGSCEflowPro)     delete theQGSCEflowPro;
  if (theQGSCPro)          delete theQGSCPro;
  if (theQGSPPro)          delete theQGSPPro; 
  if (theFTFPro)           delete theFTFPro;
  if (theRPGPro)           delete theRPGPro;
  delete thePro;
  if (theBertiniPiK)       delete theBertiniPiK;
  if (theBinaryPiK)        delete theBinaryPiK;
  if (theFTFCPiK)          delete theFTFCPiK;
  if (theFTFPPiK)          delete theFTFPPiK;
  if (theLEPPiK)           delete theLEPPiK;
  if (theLHEPPiK)          delete theLHEPPiK;
  if (theQGSCEflowPiK)     delete theQGSCEflowPiK;
  if (theQGSPPiK)          delete theQGSPPiK;
  if (theFTFPiK)           delete theFTFPiK;
  if (theRPGPiK)           delete theRPGPiK;
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
  theMiscLHEP->Build();
}

