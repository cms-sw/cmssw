#include "SimG4Core/PhysicsLists/interface/HadronPhysicsQGSPCMS_FTFP_BERT.h"
#include "SimG4Core/PhysicsLists/interface/CMSHyperonFTFPBuilder.h"

#include "globals.hh"
#include "G4ios.hh"
#include "G4SystemOfUnits.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"

#include "G4PionBuilder.hh"
#include "G4FTFPPionBuilder.hh"
#include "G4QGSPPionBuilder.hh"
#include "G4BertiniPionBuilder.hh"

#include "G4KaonBuilder.hh"
#include "G4FTFPKaonBuilder.hh"
#include "G4QGSPKaonBuilder.hh"
#include "G4BertiniKaonBuilder.hh"

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
#include "G4MesonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4ShortLivedConstructor.hh"
#include "G4IonConstructor.hh"

#include "G4HadronCaptureProcess.hh"
#include "G4NeutronRadCapture.hh"
#include "G4NeutronInelasticXS.hh"
#include "G4NeutronCaptureXS.hh"
#include "G4CrossSectionDataSetRegistry.hh"

#include "G4PhysListUtil.hh"
#include "G4ProcessManager.hh"

HadronPhysicsQGSPCMS_FTFP_BERT::HadronPhysicsQGSPCMS_FTFP_BERT(G4int)
    : HadronPhysicsQGSPCMS_FTFP_BERT(3., 6., 12., 25., 12.) {}

HadronPhysicsQGSPCMS_FTFP_BERT::HadronPhysicsQGSPCMS_FTFP_BERT(
    G4double e1, G4double e2, G4double e3, G4double e4, G4double e5)
    : G4VPhysicsConstructor("hInelasticQGSPCMS_FTFP_BERT") {
  minFTFP_ = e1;
  maxBERT_ = e2;
  minQGSP_ = e3;
  maxFTFP_ = e4;
  maxBERTpi_ = e5;
}

HadronPhysicsQGSPCMS_FTFP_BERT::~HadronPhysicsQGSPCMS_FTFP_BERT() {}

void HadronPhysicsQGSPCMS_FTFP_BERT::ConstructParticle() {
  G4MesonConstructor pMesonConstructor;
  pMesonConstructor.ConstructParticle();

  G4BaryonConstructor pBaryonConstructor;
  pBaryonConstructor.ConstructParticle();

  G4ShortLivedConstructor pShortLivedConstructor;
  pShortLivedConstructor.ConstructParticle();
}

void HadronPhysicsQGSPCMS_FTFP_BERT::DumpBanner() {
  G4cout << "### QGSP_FTFP_BERT : transition between BERT and FTFP is over the interval " << minFTFP_ / CLHEP::GeV
         << " to " << maxBERT_ / CLHEP::GeV << " GeV"
         << "                     transition between FTFP and QGSP is over the interval " << minQGSP_ / CLHEP::GeV
         << " to " << maxFTFP_ / CLHEP::GeV << G4endl;
}

void HadronPhysicsQGSPCMS_FTFP_BERT::CreateModels() {
  Neutron();
  Proton();
  Pion();
  Kaon();
  Others();
}

void HadronPhysicsQGSPCMS_FTFP_BERT::Neutron() {
  //General schema:
  // 1) Create a builder
  // 2) Call AddBuilder
  // 3) Configure the builder, possibly with sub-builders
  // 4) Call builder->Build()
  auto neu = new G4NeutronBuilder;
  AddBuilder(neu);
  auto qgs = new G4QGSPNeutronBuilder(true);
  AddBuilder(qgs);
  qgs->SetMinEnergy(minQGSP_);
  neu->RegisterMe(qgs);
  auto ftf = new G4FTFPNeutronBuilder(false);
  AddBuilder(ftf);
  ftf->SetMinEnergy(minFTFP_);
  ftf->SetMaxEnergy(maxFTFP_);
  neu->RegisterMe(ftf);
  auto bert = new G4BertiniNeutronBuilder;
  AddBuilder(bert);
  bert->SetMinEnergy(0.0);
  bert->SetMaxEnergy(maxBERT_);
  neu->RegisterMe(bert);
  neu->Build();
}

void HadronPhysicsQGSPCMS_FTFP_BERT::Proton() {
  auto pro = new G4ProtonBuilder;
  AddBuilder(pro);
  auto qgs = new G4QGSPProtonBuilder(true);
  AddBuilder(qgs);
  qgs->SetMinEnergy(minQGSP_);
  pro->RegisterMe(qgs);
  auto ftf = new G4FTFPProtonBuilder(false);
  AddBuilder(ftf);
  ftf->SetMinEnergy(minFTFP_);
  ftf->SetMaxEnergy(maxFTFP_);
  pro->RegisterMe(ftf);
  auto bert = new G4BertiniProtonBuilder;
  AddBuilder(bert);
  bert->SetMinEnergy(0.0);
  bert->SetMaxEnergy(maxBERT_);
  pro->RegisterMe(bert);
  pro->Build();
}

void HadronPhysicsQGSPCMS_FTFP_BERT::Pion() {
  auto pi = new G4PionBuilder;
  AddBuilder(pi);
  auto qgs = new G4QGSPPionBuilder(true);
  AddBuilder(qgs);
  qgs->SetMinEnergy(minQGSP_);
  pi->RegisterMe(qgs);
  auto ftf = new G4FTFPPionBuilder(false);
  AddBuilder(ftf);
  ftf->SetMinEnergy(minFTFP_);
  ftf->SetMaxEnergy(maxFTFP_);
  pi->RegisterMe(ftf);
  auto bert = new G4BertiniPionBuilder;
  AddBuilder(bert);
  bert->SetMinEnergy(0.0);
  bert->SetMaxEnergy(maxBERTpi_);
  pi->RegisterMe(bert);
  pi->Build();
}

void HadronPhysicsQGSPCMS_FTFP_BERT::Kaon() {
  auto k = new G4KaonBuilder;
  AddBuilder(k);
  auto qgs = new G4QGSPKaonBuilder(true);
  AddBuilder(qgs);
  qgs->SetMinEnergy(minQGSP_);
  k->RegisterMe(qgs);
  auto ftf = new G4FTFPKaonBuilder(false);
  AddBuilder(ftf);
  k->RegisterMe(ftf);
  ftf->SetMinEnergy(minFTFP_);
  ftf->SetMaxEnergy(maxFTFP_);
  auto bert = new G4BertiniKaonBuilder;
  AddBuilder(bert);
  k->RegisterMe(bert);
  bert->SetMaxEnergy(maxBERT_);
  bert->SetMinEnergy(0.0);
  k->Build();
}

void HadronPhysicsQGSPCMS_FTFP_BERT::Others() {
  auto hyp = new CMSHyperonFTFPBuilder();
  AddBuilder(hyp);
  hyp->Build();

  auto abar = new G4AntiBarionBuilder();
  AddBuilder(abar);
  auto ftf = new G4FTFPAntiBarionBuilder(false);
  AddBuilder(ftf);
  abar->RegisterMe(ftf);
  abar->Build();
}

void HadronPhysicsQGSPCMS_FTFP_BERT::ConstructProcess() {
  if (G4Threading::IsMasterThread()) {
    DumpBanner();
  }
  CreateModels();
  ExtraConfiguration();
}

void HadronPhysicsQGSPCMS_FTFP_BERT::ExtraConfiguration() {
  const G4ParticleDefinition* neutron = G4Neutron::Neutron();
  G4HadronicProcess* inel = G4PhysListUtil::FindInelasticProcess(neutron);
  if (inel) {
    inel->AddDataSet(new G4NeutronInelasticXS());
  }

  G4HadronicProcess* capture = nullptr;
  G4ProcessVector* pvec = neutron->GetProcessManager()->GetProcessList();
  size_t n = pvec->size();
  for (size_t i = 0; i < n; ++i) {
    if (fCapture == ((*pvec)[i])->GetProcessSubType()) {
      capture = static_cast<G4HadronicProcess*>((*pvec)[i]);
      break;
    }
  }
  if (capture) {
    capture->RegisterMe(new G4NeutronRadCapture());
    capture->AddDataSet(new G4NeutronCaptureXS());
  }
}
