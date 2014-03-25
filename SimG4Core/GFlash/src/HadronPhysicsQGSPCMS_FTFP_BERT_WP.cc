#include "SimG4Core/GFlash/interface/HadronPhysicsQGSPCMS_FTFP_BERT_WP.h"
#include "SimG4Core/GFlash/interface/G4PiKBuilder_WP.h"
#include "SimG4Core/GFlash/interface/G4ProtonBuilder_WP.h"
#include "SimG4Core/GFlash/interface/G4MiscLHEPBuilder_WP.h"

#include "globals.hh"
#include "G4ios.hh"
#include <iomanip>   
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"

#include "G4MesonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4ShortLivedConstructor.hh"

#include "G4SystemOfUnits.hh"

HadronPhysicsQGSPCMS_FTFP_BERT_WP::HadronPhysicsQGSPCMS_FTFP_BERT_WP(const G4String& name, G4bool quasiElastic)
                    :  G4VPhysicsConstructor(name) , QuasiElastic(quasiElastic)
{
   ProjectileDiffraction=false;
}

void HadronPhysicsQGSPCMS_FTFP_BERT_WP::CreateModels()
{
  // First transition, between BERT and FTF/P
  G4double minFTFP= 6.0 * GeV;     // Was 9.5 for LEP   (in FTFP_BERT 6.0 * GeV);
  G4double maxBERT= 8.0 * GeV;     // Was 9.9 for LEP   (in FTFP_BERT 8.0 * GeV);
  // Second transition, between FTF/P and QGS/P
  G4double minQGSP= 12.0 * GeV;
  G4double maxFTFP= 25.0 * GeV; 

  G4bool   quasiElasFTF= false;   // Use built-in quasi-elastic (not add-on)
  G4bool   quasiElasQGS= true;    // For QGS, it must use it.

  G4cout << " New QGSPCMS_FTFP_BERT physics list, replaces LEP with FTF/P for p/n/pi (/K?)";
  G4cout << "  Thresholds: " << G4endl;
  G4cout << "    1) between BERT  and FTF/P over the interval " 
	 << minFTFP/GeV << " to " << maxBERT/GeV << " GeV. " << G4endl;
  G4cout << "    2) between FTF/P and QGS/P over the interval " 
	 << minQGSP/GeV << " to " << maxFTFP/GeV << " GeV. " << G4endl;
  G4cout << "  -- quasiElastic was asked to be " << QuasiElastic << G4endl
	 << "     Changed to " << quasiElasQGS << " for QGS "
	 << " and to " << quasiElasFTF << " (must be false) for FTF" << G4endl;

  theNeutrons=new G4NeutronBuilder;
  theNeutrons->RegisterMe(theQGSPNeutron=new G4QGSPNeutronBuilder(quasiElasQGS));
  theQGSPNeutron->SetMinEnergy(minQGSP);   
  theNeutrons->RegisterMe(theFTFPNeutron=new CMSFTFPNeutronBuilder(quasiElasFTF));
  theFTFPNeutron->SetMinEnergy(minFTFP);   // was (9.5*GeV);
  theFTFPNeutron->SetMaxEnergy(maxFTFP);   // was (25*GeV);  
  // Exclude LEP only from Inelastic 
  //  -- Register it for other processes: Capture, Elastic
  //theNeutrons->RegisterMe(theLEPNeutron=new G4LEPNeutronBuilder);
  //theLEPNeutron->SetMinInelasticEnergy(0.0*GeV);
  // theLEPNeutron->SetMaxInelasticEnergy(0.0*GeV);

  theNeutrons->RegisterMe(theBertiniNeutron=new G4BertiniNeutronBuilder);
  theBertiniNeutron->SetMinEnergy(0.0*GeV);
  theBertiniNeutron->SetMaxEnergy(maxBERT);         // was (9.9*GeV);

  thePro=new G4ProtonBuilder_WP;
  thePro->RegisterMe(theQGSPPro=new G4QGSPProtonBuilder(quasiElasQGS));
  theQGSPPro->SetMinEnergy(minQGSP);   
  thePro->RegisterMe(theFTFPPro=new CMSFTFPProtonBuilder(quasiElasFTF));
  theFTFPPro->SetMinEnergy(minFTFP);   // was (9.5*GeV);
  theFTFPPro->SetMaxEnergy(maxFTFP);   // was (25*GeV); 

  thePro->RegisterMe(theBertiniPro=new G4BertiniProtonBuilder);
  theBertiniPro->SetMaxEnergy(maxBERT);  //  was (9.9*GeV);
  
  thePiK=new G4PiKBuilder_WP;
  thePiK->RegisterMe(theQGSPPiK=new G4QGSPPiKBuilder(quasiElasQGS));
  theQGSPPiK->SetMinEnergy(minQGSP);   
  thePiK->RegisterMe(theFTFPPiK=new CMSFTFPPiKBuilder(quasiElasFTF));
  theFTFPPiK->SetMaxEnergy(maxFTFP);   // was (25*GeV); 
  theFTFPPiK->SetMinEnergy(minFTFP);   // was (9.5*GeV);

  thePiK->RegisterMe(theBertiniPiK=new G4BertiniPiKBuilder);
  theBertiniPiK->SetMaxEnergy(maxBERT);  //  was (9.9*GeV);
  
  theMiscLHEP=new G4MiscLHEPBuilder_WP;
}

HadronPhysicsQGSPCMS_FTFP_BERT_WP::~HadronPhysicsQGSPCMS_FTFP_BERT_WP()
{
   delete theMiscLHEP;
   delete theQGSPNeutron;
   delete theFTFPNeutron;
   delete theBertiniNeutron;
   delete theQGSPPro;
   delete theFTFPPro;
   delete thePro;
   delete theBertiniPro;
   delete theQGSPPiK;
   delete theFTFPPiK;
   delete theBertiniPiK;
   delete thePiK;
}

void HadronPhysicsQGSPCMS_FTFP_BERT_WP::ConstructParticle()
{
  G4MesonConstructor pMesonConstructor;
  pMesonConstructor.ConstructParticle();

  G4BaryonConstructor pBaryonConstructor;
  pBaryonConstructor.ConstructParticle();

  G4ShortLivedConstructor pShortLivedConstructor;
  pShortLivedConstructor.ConstructParticle();  
}

#include "G4ProcessManager.hh"
void HadronPhysicsQGSPCMS_FTFP_BERT_WP::ConstructProcess()
{
  CreateModels();
  theNeutrons->Build();
  thePro->Build();
  thePiK->Build();
  theMiscLHEP->Build();
}

