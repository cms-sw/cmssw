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
#include <iomanip>   

#include "SimG4Core/PhysicsLists/interface/CMSHadronPhysicsFTFP_BERT_ATL.h"

#include "globals.hh"
#include "G4ios.hh"
#include "G4SystemOfUnits.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"

#include "G4MesonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4ShortLivedConstructor.hh"

#include "G4ComponentGGHadronNucleusXsc.hh"
#include "G4CrossSectionInelastic.hh"
#include "G4HadronCaptureProcess.hh"
#include "G4NeutronRadCapture.hh"
#include "G4NeutronInelasticXS.hh"
#include "G4NeutronCaptureXS.hh"

#include "G4CrossSectionDataSetRegistry.hh"

#include "G4PhysListUtil.hh"

G4ThreadLocal CMSHadronPhysicsFTFP_BERT_ATL::ThreadPrivate* CMSHadronPhysicsFTFP_BERT_ATL::tpdata=nullptr;

CMSHadronPhysicsFTFP_BERT_ATL::CMSHadronPhysicsFTFP_BERT_ATL(G4int)
    :  G4VPhysicsConstructor("hInelastic FTFP_BERT_ATL")
    , QuasiElastic(false)
{}

void CMSHadronPhysicsFTFP_BERT_ATL::CreateModels()
{
  G4double minFTFP =  9.0 * GeV;
  G4double maxBERT = 12.0 * GeV;
  G4cout << " CMS_FTFP_BERT_ATL : new threshold between BERT and FTFP" 
         << " is over the interval " << minFTFP/GeV << " to " << maxBERT/GeV 
         << " GeV." << G4endl;
  QuasiElastic= false; 

  tpdata->theNeutrons=new G4NeutronBuilder;
  tpdata->theNeutrons->RegisterMe(tpdata->theFTFPNeutron=new G4FTFPNeutronBuilder(QuasiElastic));
  tpdata->theFTFPNeutron->SetMinEnergy(minFTFP);
  tpdata->theNeutrons->RegisterMe(tpdata->theBertiniNeutron=new G4BertiniNeutronBuilder);
  tpdata->theBertiniNeutron->SetMinEnergy(0.0*GeV);
  tpdata->theBertiniNeutron->SetMaxEnergy(maxBERT);

  tpdata->thePro=new G4ProtonBuilder;
  tpdata->thePro->RegisterMe(tpdata->theFTFPPro=new G4FTFPProtonBuilder(QuasiElastic));
  tpdata->theFTFPPro->SetMinEnergy(minFTFP);
  tpdata->thePro->RegisterMe(tpdata->theBertiniPro=new G4BertiniProtonBuilder);
  tpdata->theBertiniPro->SetMaxEnergy(maxBERT);

  tpdata->thePiK=new G4PiKBuilder;
  tpdata->thePiK->RegisterMe(tpdata->theFTFPPiK=new G4FTFPPiKBuilder(QuasiElastic));
  tpdata->theFTFPPiK->SetMinEnergy(minFTFP);
  tpdata->thePiK->RegisterMe(tpdata->theBertiniPiK=new G4BertiniPiKBuilder);
  tpdata->theBertiniPiK->SetMaxEnergy(maxBERT);
  
  tpdata->theHyperon=new G4HyperonFTFPBuilder;
    
  tpdata->theAntiBaryon=new G4AntiBarionBuilder;
  tpdata->theAntiBaryon->RegisterMe(tpdata->theFTFPAntiBaryon=new G4FTFPAntiBarionBuilder(QuasiElastic));
}

CMSHadronPhysicsFTFP_BERT_ATL::~CMSHadronPhysicsFTFP_BERT_ATL()
{
  if (!tpdata) return;

  delete tpdata->theNeutrons;
  delete tpdata->theBertiniNeutron;
  delete tpdata->theFTFPNeutron;

  delete tpdata->thePiK;
  delete tpdata->theBertiniPiK;
  delete tpdata->theFTFPPiK;
    
  delete tpdata->thePro;
  delete tpdata->theBertiniPro;
  delete tpdata->theFTFPPro;    
    
  delete tpdata->theHyperon;
  delete tpdata->theAntiBaryon;
  delete tpdata->theFTFPAntiBaryon;
 
  //Note that here we need to set to 0 the pointer
  //since tpdata is static and if thread are "reused"
  //it can be problematic
  delete tpdata; tpdata = nullptr;
}

void CMSHadronPhysicsFTFP_BERT_ATL::ConstructParticle()
{
  G4MesonConstructor pMesonConstructor;
  pMesonConstructor.ConstructParticle();

  G4BaryonConstructor pBaryonConstructor;
  pBaryonConstructor.ConstructParticle();

  G4ShortLivedConstructor pShortLivedConstructor;
  pShortLivedConstructor.ConstructParticle();  
}

#include "G4ProcessManager.hh"
void CMSHadronPhysicsFTFP_BERT_ATL::ConstructProcess()
{
  if ( tpdata == nullptr ) tpdata = new ThreadPrivate;
  CreateModels();
  tpdata->theNeutrons->Build();
  tpdata->thePro->Build();
  tpdata->thePiK->Build();

  // --- Kaons ---
  tpdata->xsKaon = new G4ComponentGGHadronNucleusXsc();
  G4VCrossSectionDataSet * kaonxs = new G4CrossSectionInelastic(tpdata->xsKaon);
  G4PhysListUtil::FindInelasticProcess(G4KaonMinus::KaonMinus())->AddDataSet(kaonxs);
  G4PhysListUtil::FindInelasticProcess(G4KaonPlus::KaonPlus())->AddDataSet(kaonxs);
  G4PhysListUtil::FindInelasticProcess(G4KaonZeroShort::KaonZeroShort())->AddDataSet(kaonxs);
  G4PhysListUtil::FindInelasticProcess(G4KaonZeroLong::KaonZeroLong())->AddDataSet(kaonxs);
    
  tpdata->theHyperon->Build();
  tpdata->theAntiBaryon->Build();

  // --- Neutrons ---
  tpdata->xsNeutronInelasticXS = new G4NeutronInelasticXS();

  G4HadronicProcess* capture = nullptr;
  G4ProcessManager* pmanager = G4Neutron::Neutron()->GetProcessManager();
  G4ProcessVector*  pv = pmanager->GetProcessList();
  for ( size_t i=0; i < static_cast<size_t>(pv->size()); ++i ) {
    if ( fCapture == ((*pv)[i])->GetProcessSubType() ) {
      capture = static_cast<G4HadronicProcess*>((*pv)[i]);
    }
  }
  if ( ! capture ) {
    capture = new G4HadronCaptureProcess("nCapture");
    pmanager->AddDiscreteProcess(capture);
  }
  tpdata->xsNeutronCaptureXS = new G4NeutronCaptureXS();
  capture->AddDataSet(tpdata->xsNeutronCaptureXS);
  capture->RegisterMe(new G4NeutronRadCapture());
}
