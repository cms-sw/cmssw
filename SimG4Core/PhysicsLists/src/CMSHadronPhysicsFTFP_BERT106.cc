
#include <iomanip>

#include "SimG4Core/PhysicsLists/interface/CMSHadronPhysicsFTFP_BERT106.h"
#include "SimG4Core/PhysicsLists/interface/CMSHyperonFTFPBuilder.h"

#include "globals.hh"
#include "G4ios.hh"
#include "G4SystemOfUnits.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4PionBuilder.hh"
#include "G4BertiniPionBuilder.hh"
#include "G4FTFPPionBuilder.hh"

#include "G4KaonBuilder.hh"
#include "G4BertiniKaonBuilder.hh"
#include "G4FTFPKaonBuilder.hh"

#include "G4ProtonBuilder.hh"
#include "G4BertiniProtonBuilder.hh"
#include "G4FTFPNeutronBuilder.hh"
#include "G4FTFPProtonBuilder.hh"

#include "G4NeutronBuilder.hh"
#include "G4BertiniNeutronBuilder.hh"
#include "G4FTFPNeutronBuilder.hh"

#include "G4HyperonFTFPBuilder.hh"
#include "G4AntiBarionBuilder.hh"
#include "G4FTFPAntiBarionBuilder.hh"

#include "G4MesonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4ShortLivedConstructor.hh"

#include "G4HadronCaptureProcess.hh"
#include "G4NeutronRadCapture.hh"
#include "G4NeutronInelasticXS.hh"
#include "G4NeutronCaptureXS.hh"

#include "G4PhysListUtil.hh"
#include "G4Threading.hh"

#include "G4DeexPrecoParameters.hh"
#include "G4NuclearLevelData.hh"

#include "G4ProcessManager.hh"

CMSHadronPhysicsFTFP_BERT106::CMSHadronPhysicsFTFP_BERT106(G4int)
    : CMSHadronPhysicsFTFP_BERT106(3. * CLHEP::GeV, 6. * CLHEP::GeV, 12 * CLHEP::GeV) {}

CMSHadronPhysicsFTFP_BERT106::CMSHadronPhysicsFTFP_BERT106(G4double e1, G4double e2, G4double e3)
    : G4VPhysicsConstructor("hInelastic FTFP_BERT") {
  minFTFP_ = e1;
  maxBERT_ = e2;
  maxBERTpi_ = e3;
}

CMSHadronPhysicsFTFP_BERT106::~CMSHadronPhysicsFTFP_BERT106() {}

void CMSHadronPhysicsFTFP_BERT106::ConstructParticle() {
  G4MesonConstructor pMesonConstructor;
  pMesonConstructor.ConstructParticle();

  G4BaryonConstructor pBaryonConstructor;
  pBaryonConstructor.ConstructParticle();

  G4ShortLivedConstructor pShortLivedConstructor;
  pShortLivedConstructor.ConstructParticle();
}

void CMSHadronPhysicsFTFP_BERT106::DumpBanner() {
  G4cout << "### FTFP_BERT : transition between BERT and FTFP is over the interval " << minFTFP_ / CLHEP::GeV << " to "
         << maxBERT_ / CLHEP::GeV << " GeV"
         << " GeV; for pions up to " << maxBERTpi_ / CLHEP::GeV << " GeV" << G4endl;
}

void CMSHadronPhysicsFTFP_BERT106::CreateModels() {
  Neutron();
  Proton();
  Pion();
  Kaon();
  Others();
}

void CMSHadronPhysicsFTFP_BERT106::Neutron() {
  //General schema:
  // 1) Create a builder
  // 2) Call AddBuilder
  // 3) Configure the builder, possibly with sub-builders
  // 4) Call builder->Build()
  auto neu = new G4NeutronBuilder;
  AddBuilder(neu);
  auto ftfpn = new G4FTFPNeutronBuilder(false);
  AddBuilder(ftfpn);
  neu->RegisterMe(ftfpn);
  ftfpn->SetMinEnergy(minFTFP_);
  auto bertn = new G4BertiniNeutronBuilder;
  AddBuilder(bertn);
  neu->RegisterMe(bertn);
  bertn->SetMinEnergy(0.0);
  bertn->SetMaxEnergy(maxBERT_);
  neu->Build();
}

void CMSHadronPhysicsFTFP_BERT106::Proton() {
  auto pro = new G4ProtonBuilder;
  AddBuilder(pro);
  auto ftfpp = new G4FTFPProtonBuilder(false);
  AddBuilder(ftfpp);
  pro->RegisterMe(ftfpp);
  ftfpp->SetMinEnergy(minFTFP_);
  auto bertp = new G4BertiniProtonBuilder;
  AddBuilder(bertp);
  pro->RegisterMe(bertp);
  bertp->SetMaxEnergy(maxBERT_);
  pro->Build();
}

void CMSHadronPhysicsFTFP_BERT106::Pion() {
  auto pi = new G4PionBuilder;
  AddBuilder(pi);
  auto ftfppi = new G4FTFPPionBuilder(false);
  AddBuilder(ftfppi);
  pi->RegisterMe(ftfppi);
  ftfppi->SetMinEnergy(minFTFP_);
  auto bertpi = new G4BertiniPionBuilder;
  AddBuilder(bertpi);
  pi->RegisterMe(bertpi);
  bertpi->SetMaxEnergy(maxBERTpi_);
  pi->Build();
}

void CMSHadronPhysicsFTFP_BERT106::Kaon() {
  auto k = new G4KaonBuilder;
  AddBuilder(k);
  auto ftfpk = new G4FTFPKaonBuilder(false);
  AddBuilder(ftfpk);
  k->RegisterMe(ftfpk);
  ftfpk->SetMinEnergy(minFTFP_);
  auto bertk = new G4BertiniKaonBuilder;
  AddBuilder(bertk);
  k->RegisterMe(bertk);
  bertk->SetMaxEnergy(maxBERT_);
  k->Build();
}

void CMSHadronPhysicsFTFP_BERT106::Others() {
  //===== Hyperons ====== //
  auto hyp = new CMSHyperonFTFPBuilder;
  AddBuilder(hyp);
  hyp->Build();

  ///===== Anti-barions==== //
  auto abar = new G4AntiBarionBuilder;
  AddBuilder(abar);
  auto ftfpabar = new G4FTFPAntiBarionBuilder(false);
  AddBuilder(ftfpabar);
  abar->RegisterMe(ftfpabar);
  abar->Build();
}

void CMSHadronPhysicsFTFP_BERT106::ConstructProcess() {
  if (G4Threading::IsMasterThread()) {
    DumpBanner();
  }
  CreateModels();
  ExtraConfiguration();
}

void CMSHadronPhysicsFTFP_BERT106::ExtraConfiguration() {
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
