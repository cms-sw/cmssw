#include "SimG4Core/PhysicsLists/interface/HadronPhysicsQGSPCMS_FTFP_BERT.h"

#include "globals.hh"
#include "G4ios.hh"
#include <iomanip>
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"

#include "G4MesonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4ShortLivedConstructor.hh"
#include "G4IonConstructor.hh"

#include "G4HadronCaptureProcess.hh"
#include "G4NeutronRadCapture.hh"
#include "G4NeutronInelasticXS.hh"
#include "G4NeutronCaptureXS.hh"

#include "G4PhysListUtil.hh"
#include "G4SystemOfUnits.hh"

G4ThreadLocal HadronPhysicsQGSPCMS_FTFP_BERT::ThreadPrivate* HadronPhysicsQGSPCMS_FTFP_BERT::tpdata = nullptr;

HadronPhysicsQGSPCMS_FTFP_BERT::HadronPhysicsQGSPCMS_FTFP_BERT(G4int)
    : G4VPhysicsConstructor("hInelasticQGSPCMS_FTFP_BERT") {}

void HadronPhysicsQGSPCMS_FTFP_BERT::CreateModels() {
  // First transition, between BERT and FTF/P
  G4double minFTFP = 6.0 * GeV;
  G4double maxBERT = 8.0 * GeV;
  // Second transition, between FTF/P and QGS/P
  G4double minQGSP = 12.0 * GeV;
  G4double maxFTFP = 25.0 * GeV;

  G4bool quasiElasFTF = false;  // Use built-in quasi-elastic (not add-on)
  G4bool quasiElasQGS = true;   // For QGS, it must use it.

  G4cout << " New QGSPCMS_FTFP_BERT hadronic inealstic physics" << G4endl;
  G4cout << "   Thresholds: " << G4endl;
  G4cout << "     1) between BERT  and FTFP over the interval " << minFTFP / GeV << " to " << maxBERT / GeV << " GeV. "
         << G4endl;
  G4cout << "     2) between FTFP and QGSP over the interval " << minQGSP / GeV << " to " << maxFTFP / GeV << " GeV. "
         << G4endl;
  G4cout << "   QuasiElastic: " << quasiElasQGS << " for QGS "
         << " and " << quasiElasFTF << " for FTF " << G4endl;

  tpdata->theNeutrons = new G4NeutronBuilder;
  tpdata->theNeutrons->RegisterMe(tpdata->theQGSPNeutron = new G4QGSPNeutronBuilder(quasiElasQGS));
  tpdata->theQGSPNeutron->SetMinEnergy(minQGSP);
  tpdata->theNeutrons->RegisterMe(tpdata->theFTFPNeutron = new G4FTFPNeutronBuilder(quasiElasFTF));
  tpdata->theFTFPNeutron->SetMinEnergy(minFTFP);
  tpdata->theFTFPNeutron->SetMaxEnergy(maxFTFP);

  tpdata->theNeutrons->RegisterMe(tpdata->theBertiniNeutron = new G4BertiniNeutronBuilder);
  tpdata->theBertiniNeutron->SetMinEnergy(0.0 * GeV);
  tpdata->theBertiniNeutron->SetMaxEnergy(maxBERT);

  tpdata->thePro = new G4ProtonBuilder;
  tpdata->thePro->RegisterMe(tpdata->theQGSPPro = new G4QGSPProtonBuilder(quasiElasQGS));
  tpdata->theQGSPPro->SetMinEnergy(minQGSP);
  tpdata->thePro->RegisterMe(tpdata->theFTFPPro = new G4FTFPProtonBuilder(quasiElasFTF));
  tpdata->theFTFPPro->SetMinEnergy(minFTFP);
  tpdata->theFTFPPro->SetMaxEnergy(maxFTFP);
  tpdata->thePro->RegisterMe(tpdata->theBertiniPro = new G4BertiniProtonBuilder);
  tpdata->theBertiniPro->SetMaxEnergy(maxBERT);

  tpdata->thePiK = new G4PiKBuilder;
  tpdata->thePiK->RegisterMe(tpdata->theQGSPPiK = new G4QGSPPiKBuilder(quasiElasQGS));
  tpdata->theQGSPPiK->SetMinEnergy(minQGSP);
  tpdata->thePiK->RegisterMe(tpdata->theFTFPPiK = new G4FTFPPiKBuilder(quasiElasFTF));
  tpdata->theFTFPPiK->SetMaxEnergy(maxFTFP);
  tpdata->theFTFPPiK->SetMinEnergy(minFTFP);
  tpdata->thePiK->RegisterMe(tpdata->theBertiniPiK = new G4BertiniPiKBuilder);
  tpdata->theBertiniPiK->SetMaxEnergy(maxBERT);

  // Hyperons use FTF
  tpdata->theHyperon = new G4HyperonFTFPBuilder;

  tpdata->theAntiBaryon = new G4AntiBarionBuilder;
  tpdata->theAntiBaryon->RegisterMe(tpdata->theFTFPAntiBaryon = new G4FTFPAntiBarionBuilder(quasiElasFTF));
}

HadronPhysicsQGSPCMS_FTFP_BERT::~HadronPhysicsQGSPCMS_FTFP_BERT() {
  if (nullptr != tpdata) {
    delete tpdata->theQGSPNeutron;
    delete tpdata->theFTFPNeutron;
    delete tpdata->theBertiniNeutron;
    delete tpdata->theNeutrons;

    delete tpdata->theQGSPPro;
    delete tpdata->theFTFPPro;
    delete tpdata->thePro;
    delete tpdata->theBertiniPro;

    delete tpdata->theQGSPPiK;
    delete tpdata->theFTFPPiK;
    delete tpdata->theBertiniPiK;
    delete tpdata->thePiK;

    delete tpdata->theHyperon;
    delete tpdata->theAntiBaryon;
    delete tpdata->theFTFPAntiBaryon;

    delete tpdata;
    tpdata = nullptr;
  }
}

void HadronPhysicsQGSPCMS_FTFP_BERT::ConstructParticle() {
  G4MesonConstructor pMesonConstructor;
  pMesonConstructor.ConstructParticle();

  G4BaryonConstructor pBaryonConstructor;
  pBaryonConstructor.ConstructParticle();

  G4ShortLivedConstructor pShortLivedConstructor;
  pShortLivedConstructor.ConstructParticle();

  G4IonConstructor pIonConstructor;
  pIonConstructor.ConstructParticle();
}

#include "G4ProcessManager.hh"
void HadronPhysicsQGSPCMS_FTFP_BERT::ConstructProcess() {
  if (tpdata == nullptr) {
    tpdata = new ThreadPrivate;
  }
  CreateModels();
  tpdata->theNeutrons->Build();
  tpdata->thePro->Build();
  tpdata->thePiK->Build();
  tpdata->theHyperon->Build();
  tpdata->theAntiBaryon->Build();

  // --- Neutrons ---
  G4PhysListUtil::FindInelasticProcess(G4Neutron::Neutron())->AddDataSet(new G4NeutronInelasticXS());

  G4HadronicProcess* capture = nullptr;
  G4ProcessManager* pmanager = G4Neutron::Neutron()->GetProcessManager();
  G4ProcessVector* pv = pmanager->GetProcessList();
  for (size_t i = 0; i < static_cast<size_t>(pv->size()); ++i) {
    if (fCapture == ((*pv)[i])->GetProcessSubType()) {
      capture = static_cast<G4HadronicProcess*>((*pv)[i]);
    }
  }
  if (!capture) {
    capture = new G4HadronCaptureProcess("nCapture");
    pmanager->AddDiscreteProcess(capture);
  }
  capture->AddDataSet(new G4NeutronCaptureXS());
  capture->RegisterMe(new G4NeutronRadCapture());
}
