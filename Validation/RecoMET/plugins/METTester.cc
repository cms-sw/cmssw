// author: Mike Schmitt, University of Florida
// first version 8/24/2006
// modification: Bobby Scurlock
// date:  03.11.2006
// note:  added RMS(METx) vs SumET capability
// modification: Rick Cavanaugh
// date:  05.11.2006
// note:  cleaned up constructor and beginJob, removed int conv. warning
//        added configuration params
// modification: Mike Schmitt
// date:  02.28.2007
// note:  code rewrite. Now uses STL map for monitoring element container.
// modification: Bobby Scurlock
// date:  04.03.2007
// note:  Eliminated automated resolution fitting. This is now done in a ROOT
// script.

// date:  02.04.2009
// note:  Added option to use fine binning or course binning for histos
//
// modification: Samantha Hewamanage, Florida International University
// date: 01.30.2012
// note: Added few hists for various nvtx ranges to study PU effects.
//       Cleaned up the code by making it readable and const'ing the
//       variables that should be changed.
//       Changed the number of bins from odd to even. Odd number of bins
//       makes it impossible to rebin a hist.
#include "METTester.h"
using namespace reco;
using namespace std;
using namespace edm;

METTester::METTester(const edm::ParameterSet &iConfig) {
  inputMETLabel_ = iConfig.getParameter<edm::InputTag>("InputMETLabel");
  METType_ = iConfig.getUntrackedParameter<std::string>("METType");

  std::string inputMETCollectionLabel(inputMETLabel_.label());

  isCaloMET = (std::string("calo") == METType_);
  isPFMET = (std::string("pf") == METType_);
  isGenMET = (std::string("gen") == METType_);
  isMiniAODMET = (std::string("miniaod") == METType_);

  pvToken_ = consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("PrimaryVertices"));
  if (isCaloMET)
    caloMETsToken_ = consumes<reco::CaloMETCollection>(inputMETLabel_);
  if (isPFMET)
    pfMETsToken_ = consumes<reco::PFMETCollection>(inputMETLabel_);
  if (isMiniAODMET)
    patMETToken_ = consumes<pat::METCollection>(inputMETLabel_);
  if (isGenMET)
    genMETsToken_ = consumes<reco::GenMETCollection>(inputMETLabel_);
  if (!isMiniAODMET) {
    genMETsTrueToken_ = consumes<reco::GenMETCollection>(edm::InputTag("genMetTrue"));
    genMETsCaloToken_ = consumes<reco::GenMETCollection>(edm::InputTag("genMetCalo"));
  }

  // Events variables
  mNvertex = nullptr;

  // Common variables
  mMEx = nullptr;
  mMEy = nullptr;
  mMETSig = nullptr;
  mMET = nullptr;
  mMETFine = nullptr;
  mMET_Nvtx = nullptr;
  mMETPhi = nullptr;
  mSumET = nullptr;
  mMETDifference_GenMETTrue = nullptr;
  mMETDeltaPhi_GenMETTrue = nullptr;
  mMETDifference_GenMETCalo = nullptr;
  mMETDeltaPhi_GenMETCalo = nullptr;

  // MET Uncertainities: Only for MiniAOD
  mMETUnc_JetResUp = nullptr;
  mMETUnc_JetResDown = nullptr;
  mMETUnc_JetEnUp = nullptr;
  mMETUnc_JetEnDown = nullptr;
  mMETUnc_MuonEnUp = nullptr;
  mMETUnc_MuonEnDown = nullptr;
  mMETUnc_ElectronEnUp = nullptr;
  mMETUnc_ElectronEnDown = nullptr;
  mMETUnc_TauEnUp = nullptr;
  mMETUnc_TauEnDown = nullptr;
  mMETUnc_UnclusteredEnUp = nullptr;
  mMETUnc_UnclusteredEnDown = nullptr;
  mMETUnc_PhotonEnUp = nullptr;
  mMETUnc_PhotonEnDown = nullptr;

  // CaloMET variables
  mCaloMaxEtInEmTowers = nullptr;
  mCaloMaxEtInHadTowers = nullptr;
  mCaloEtFractionHadronic = nullptr;
  mCaloEmEtFraction = nullptr;
  mCaloHadEtInHB = nullptr;
  mCaloHadEtInHO = nullptr;
  mCaloHadEtInHE = nullptr;
  mCaloHadEtInHF = nullptr;
  mCaloEmEtInHF = nullptr;
  mCaloSETInpHF = nullptr;
  mCaloSETInmHF = nullptr;
  mCaloEmEtInEE = nullptr;
  mCaloEmEtInEB = nullptr;

  // GenMET variables
  mNeutralEMEtFraction = nullptr;
  mNeutralHadEtFraction = nullptr;
  mChargedEMEtFraction = nullptr;
  mChargedHadEtFraction = nullptr;
  mMuonEtFraction = nullptr;
  mInvisibleEtFraction = nullptr;

  // MET variables

  // PFMET variables
  mMETDifference_GenMETTrue_MET0to20 = nullptr;
  mMETDifference_GenMETTrue_MET20to40 = nullptr;
  mMETDifference_GenMETTrue_MET40to60 = nullptr;
  mMETDifference_GenMETTrue_MET60to80 = nullptr;
  mMETDifference_GenMETTrue_MET80to100 = nullptr;
  mMETDifference_GenMETTrue_MET100to150 = nullptr;
  mMETDifference_GenMETTrue_MET150to200 = nullptr;
  mMETDifference_GenMETTrue_MET200to300 = nullptr;
  mMETDifference_GenMETTrue_MET300to400 = nullptr;
  mMETDifference_GenMETTrue_MET400to500 = nullptr;
  mMETDifference_GenMETTrue_MET500 = nullptr;
}
void METTester::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const &iRun, edm::EventSetup const & /* iSetup */) {
  ibooker.setCurrentFolder("JetMET/METValidation/" + inputMETLabel_.label());

  mNvertex = ibooker.book1D("Nvertex", "Nvertex", 80, 0, 80);
  mMEx = ibooker.book1D("MEx", "MEx", 160, -800, 800);
  mMEy = ibooker.book1D("MEy", "MEy", 160, -800, 800);
  mMETSig = ibooker.book1D("METSig", "METSig", 25, 0, 24.5);
  mMET = ibooker.book1D("MET", "MET (20 GeV binning)", 100, 0, 2000);
  mMETFine = ibooker.book1D("METFine", "MET (2 GeV binning)", 1000, 0, 2000);
  mMET_Nvtx = ibooker.bookProfile("MET_Nvtx", "MET vs. nvtx", 60, 0., 60., 0., 2000., " ");
  mMETPhi = ibooker.book1D("METPhi", "METPhi", 80, -4, 4);
  mSumET = ibooker.book1D("SumET", "SumET", 200, 0, 4000);  // 10GeV
  mMETDifference_GenMETTrue = ibooker.book1D("METDifference_GenMETTrue", "METDifference_GenMETTrue", 500, -500, 500);
  mMETDeltaPhi_GenMETTrue = ibooker.book1D("METDeltaPhi_GenMETTrue", "METDeltaPhi_GenMETTrue", 80, 0, 4);

  if (isMiniAODMET) {
    mMETUnc_JetResUp = ibooker.book1D("METUnc_JetResUp", "METUnc_JetResUp", 200, -10, 10);
    mMETUnc_JetResDown = ibooker.book1D("METUnc_JetResDown", "METUnc_JetResDown", 200, -10, 10);
    mMETUnc_JetEnUp = ibooker.book1D("METUnc_JetEnUp", "METUnc_JetEnUp", 200, -10, 10);
    mMETUnc_JetEnDown = ibooker.book1D("METUnc_JetEnDown", "METUnc_JetEnDown", 200, -10, 10);
    mMETUnc_MuonEnUp = ibooker.book1D("METUnc_MuonEnUp", "METUnc_MuonEnUp", 200, -10, 10);
    mMETUnc_MuonEnDown = ibooker.book1D("METUnc_MuonEnDown", "METUnc_MuonEnDown", 200, -10, 10);
    mMETUnc_ElectronEnUp = ibooker.book1D("METUnc_ElectronEnUp", "METUnc_ElectronEnUp", 200, -10, 10);
    mMETUnc_ElectronEnDown = ibooker.book1D("METUnc_ElectronEnDown", "METUnc_ElectronEnDown", 200, -10, 10);
    mMETUnc_TauEnUp = ibooker.book1D("METUnc_TauEnUp", "METUnc_TauEnUp", 200, -10, 10);
    mMETUnc_TauEnDown = ibooker.book1D("METUnc_TauEnDown", "METUnc_TauEnDown", 200, -10, 10);
    mMETUnc_UnclusteredEnUp = ibooker.book1D("METUnc_UnclusteredEnUp", "METUnc_UnclusteredEnUp", 200, -10, 10);
    mMETUnc_UnclusteredEnDown = ibooker.book1D("METUnc_UnclusteredEnDown", "METUnc_UnclusteredEnDown", 200, -10, 10);
    mMETUnc_PhotonEnUp = ibooker.book1D("METUnc_UnclusteredEnDown", "METUnc_UnclusteredEnDown", 200, -10, 10);
    mMETUnc_PhotonEnDown = ibooker.book1D("METUnc_PhotonEnDown", "METUnc_PhotonEnDown", 200, -10, 10);
  }
  if (!isMiniAODMET) {
    mMETDifference_GenMETCalo = ibooker.book1D("METDifference_GenMETCalo", "METDifference_GenMETCalo", 500, -500, 500);
    mMETDeltaPhi_GenMETCalo = ibooker.book1D("METDeltaPhi_GenMETCalo", "METDeltaPhi_GenMETCalo", 80, 0, 4);
  }
  if (!isGenMET) {
    mMETDifference_GenMETTrue_MET0to20 =
        ibooker.book1D("METResolution_GenMETTrue_MET0to20", "METResolution_GenMETTrue_MET0to20", 500, -500, 500);
    mMETDifference_GenMETTrue_MET20to40 =
        ibooker.book1D("METResolution_GenMETTrue_MET20to40", "METResolution_GenMETTrue_MET20to40", 500, -500, 500);
    mMETDifference_GenMETTrue_MET40to60 =
        ibooker.book1D("METResolution_GenMETTrue_MET40to60", "METResolution_GenMETTrue_MET40to60", 500, -500, 500);
    mMETDifference_GenMETTrue_MET60to80 =
        ibooker.book1D("METResolution_GenMETTrue_MET60to80", "METResolution_GenMETTrue_MET60to80", 500, -500, 500);
    mMETDifference_GenMETTrue_MET80to100 =
        ibooker.book1D("METResolution_GenMETTrue_MET80to100", "METResolution_GenMETTrue_MET80to100", 500, -500, 500);
    mMETDifference_GenMETTrue_MET100to150 =
        ibooker.book1D("METResolution_GenMETTrue_MET100to150", "METResolution_GenMETTrue_MET100to150", 500, -500, 500);
    mMETDifference_GenMETTrue_MET150to200 =
        ibooker.book1D("METResolution_GenMETTrue_MET150to200", "METResolution_GenMETTrue_MET150to200", 500, -500, 500);
    mMETDifference_GenMETTrue_MET200to300 =
        ibooker.book1D("METResolution_GenMETTrue_MET200to300", "METResolution_GenMETTrue_MET200to300", 500, -500, 500);
    mMETDifference_GenMETTrue_MET300to400 =
        ibooker.book1D("METResolution_GenMETTrue_MET300to400", "METResolution_GenMETTrue_MET300to400", 500, -500, 500);
    mMETDifference_GenMETTrue_MET400to500 =
        ibooker.book1D("METResolution_GenMETTrue_MET400to500", "METResolution_GenMETTrue_MET400to500", 500, -500, 500);
    mMETDifference_GenMETTrue_MET500 =
        ibooker.book1D("METResolution_GenMETTrue_MET500", "METResolution_GenMETTrue_MET500", 500, -500, 500);
  }
  if (isCaloMET) {
    mCaloMaxEtInEmTowers = ibooker.book1D("CaloMaxEtInEmTowers", "CaloMaxEtInEmTowers", 300, 0, 1500);     // 5GeV
    mCaloMaxEtInHadTowers = ibooker.book1D("CaloMaxEtInHadTowers", "CaloMaxEtInHadTowers", 300, 0, 1500);  // 5GeV
    mCaloEtFractionHadronic = ibooker.book1D("CaloEtFractionHadronic", "CaloEtFractionHadronic", 100, 0, 1);
    mCaloEmEtFraction = ibooker.book1D("CaloEmEtFraction", "CaloEmEtFraction", 100, 0, 1);
    mCaloHadEtInHB = ibooker.book1D("CaloHadEtInHB", "CaloHadEtInHB", 200, 0, 2000);  // 5GeV
    mCaloHadEtInHE = ibooker.book1D("CaloHadEtInHE", "CaloHadEtInHE", 100, 0, 500);   // 5GeV
    mCaloHadEtInHO = ibooker.book1D("CaloHadEtInHO", "CaloHadEtInHO", 100, 0, 200);   // 5GeV
    mCaloHadEtInHF = ibooker.book1D("CaloHadEtInHF", "CaloHadEtInHF", 100, 0, 200);   // 5GeV
    mCaloSETInpHF = ibooker.book1D("CaloSETInpHF", "CaloSETInpHF", 100, 0, 500);
    mCaloSETInmHF = ibooker.book1D("CaloSETInmHF", "CaloSETInmHF", 100, 0, 500);
    mCaloEmEtInEE = ibooker.book1D("CaloEmEtInEE", "CaloEmEtInEE", 100, 0, 500);  // 5GeV
    mCaloEmEtInEB = ibooker.book1D("CaloEmEtInEB", "CaloEmEtInEB", 100, 0, 500);  // 5GeV
    mCaloEmEtInHF = ibooker.book1D("CaloEmEtInHF", "CaloEmEtInHF", 100, 0, 500);  // 5GeV
  }

  if (isGenMET) {
    mNeutralEMEtFraction = ibooker.book1D("GenNeutralEMEtFraction", "GenNeutralEMEtFraction", 120, 0.0, 1.2);
    mNeutralHadEtFraction = ibooker.book1D("GenNeutralHadEtFraction", "GenNeutralHadEtFraction", 120, 0.0, 1.2);
    mChargedEMEtFraction = ibooker.book1D("GenChargedEMEtFraction", "GenChargedEMEtFraction", 120, 0.0, 1.2);
    mChargedHadEtFraction = ibooker.book1D("GenChargedHadEtFraction", "GenChargedHadEtFraction", 120, 0.0, 1.2);
    mMuonEtFraction = ibooker.book1D("GenMuonEtFraction", "GenMuonEtFraction", 120, 0.0, 1.2);
    mInvisibleEtFraction = ibooker.book1D("GenInvisibleEtFraction", "GenInvisibleEtFraction", 120, 0.0, 1.2);
  }

  if (isPFMET || isMiniAODMET) {
    mPFphotonEtFraction = ibooker.book1D("photonEtFraction", "photonEtFraction", 100, 0, 1);
    mPFneutralHadronEtFraction = ibooker.book1D("neutralHadronEtFraction", "neutralHadronEtFraction", 100, 0, 1);
    mPFelectronEtFraction = ibooker.book1D("electronEtFraction", "electronEtFraction", 100, 0, 1);
    mPFchargedHadronEtFraction = ibooker.book1D("chargedHadronEtFraction", "chargedHadronEtFraction", 100, 0, 1);
    mPFHFHadronEtFraction = ibooker.book1D("HFHadronEtFraction", "HFHadronEtFraction", 100, 0, 1);
    mPFmuonEtFraction = ibooker.book1D("muonEtFraction", "muonEtFraction", 100, 0, 1);
    mPFHFEMEtFraction = ibooker.book1D("HFEMEtFraction", "HFEMEtFraction", 100, 0, 1);

    if (!isMiniAODMET) {
      mPFphotonEt = ibooker.book1D("photonEt", "photonEt", 100, 0, 1000);
      mPFneutralHadronEt = ibooker.book1D("neutralHadronEt", "neutralHadronEt", 100, 0, 1000);
      mPFelectronEt = ibooker.book1D("electronEt", "electronEt", 100, 0, 1000);
      mPFchargedHadronEt = ibooker.book1D("chargedHadronEt", "chargedHadronEt", 100, 0, 1000);
      mPFmuonEt = ibooker.book1D("muonEt", "muonEt", 100, 0, 1000);
      mPFHFHadronEt = ibooker.book1D("HFHadronEt", "HFHadronEt", 100, 0, 500);
      mPFHFEMEt = ibooker.book1D("HFEMEt", "HFEMEt", 100, 0, 300);
    }
  }
}

void METTester::analyze(const edm::Event &iEvent,
                        const edm::EventSetup &iSetup) {  // int counter(0);

  edm::Handle<reco::VertexCollection> pvHandle;
  iEvent.getByToken(pvToken_, pvHandle);
  if (!pvHandle.isValid()) {
    std::cout << __FUNCTION__ << ":" << __LINE__ << ":pvHandle handle not found!" << std::endl;
    assert(false);
  }
  const int nvtx = pvHandle->size();
  mNvertex->Fill(nvtx);
  // Collections for all MET collections

  edm::Handle<CaloMETCollection> caloMETs;
  edm::Handle<PFMETCollection> pfMETs;
  edm::Handle<GenMETCollection> genMETs;
  edm::Handle<pat::METCollection> patMET;

  if (isCaloMET)
    iEvent.getByToken(caloMETsToken_, caloMETs);
  if (isPFMET)
    iEvent.getByToken(pfMETsToken_, pfMETs);
  if (isGenMET)
    iEvent.getByToken(genMETsToken_, genMETs);
  if (isMiniAODMET)
    iEvent.getByToken(patMETToken_, patMET);
  if ((isCaloMET) and !caloMETs.isValid())
    return;
  if ((isPFMET) and !pfMETs.isValid())
    return;
  if ((isGenMET) and !genMETs.isValid())
    return;
  if ((isMiniAODMET) and !patMET.isValid())
    return;

  reco::MET met;
  if (isCaloMET) {
    met = caloMETs->front();
  }
  if (isPFMET) {
    met = pfMETs->front();
  }
  if (isGenMET) {
    met = genMETs->front();
  }
  if (isMiniAODMET) {
    met = patMET->front();
  }

  const double SumET = met.sumEt();
  const double METSig = met.mEtSig();
  const double MET = met.pt();
  const double MEx = met.px();
  const double MEy = met.py();
  const double METPhi = met.phi();
  mMEx->Fill(MEx);
  mMEy->Fill(MEy);
  mMET->Fill(MET);
  mMETFine->Fill(MET);
  mMET_Nvtx->Fill((double)nvtx, MET);
  mMETPhi->Fill(METPhi);
  mSumET->Fill(SumET);
  mMETSig->Fill(METSig);

  // Get Generated MET for Resolution plots
  const reco::GenMET *genMetTrue = nullptr;
  bool isvalidgenmet = false;

  if (!isMiniAODMET) {
    edm::Handle<GenMETCollection> genTrue;
    iEvent.getByToken(genMETsTrueToken_, genTrue);
    if (genTrue.isValid()) {
      isvalidgenmet = true;
      const GenMETCollection *genmetcol = genTrue.product();
      genMetTrue = &(genmetcol->front());
    }
  } else {
    genMetTrue = patMET->front().genMET();
    isvalidgenmet = true;
  }

  if (isvalidgenmet) {
    double genMET = genMetTrue->pt();
    double genMETPhi = genMetTrue->phi();

    mMETDifference_GenMETTrue->Fill(MET - genMET);
    mMETDeltaPhi_GenMETTrue->Fill(TMath::ACos(TMath::Cos(METPhi - genMETPhi)));

    if (!isGenMET) {
      // pfMET resolution in pfMET bins : Sam, Feb, 2012
      if (MET > 0 && MET < 20)
        mMETDifference_GenMETTrue_MET0to20->Fill(MET - genMET);
      else if (MET > 20 && MET < 40)
        mMETDifference_GenMETTrue_MET20to40->Fill(MET - genMET);
      else if (MET > 40 && MET < 60)
        mMETDifference_GenMETTrue_MET40to60->Fill(MET - genMET);
      else if (MET > 60 && MET < 80)
        mMETDifference_GenMETTrue_MET60to80->Fill(MET - genMET);
      else if (MET > 80 && MET < 100)
        mMETDifference_GenMETTrue_MET80to100->Fill(MET - genMET);
      else if (MET > 100 && MET < 150)
        mMETDifference_GenMETTrue_MET100to150->Fill(MET - genMET);
      else if (MET > 150 && MET < 200)
        mMETDifference_GenMETTrue_MET150to200->Fill(MET - genMET);
      else if (MET > 200 && MET < 300)
        mMETDifference_GenMETTrue_MET200to300->Fill(MET - genMET);
      else if (MET > 300 && MET < 400)
        mMETDifference_GenMETTrue_MET300to400->Fill(MET - genMET);
      else if (MET > 400 && MET < 500)
        mMETDifference_GenMETTrue_MET400to500->Fill(MET - genMET);
      else if (MET > 500)
        mMETDifference_GenMETTrue_MET500->Fill(MET - genMET);

    } else {
      edm::LogInfo("OutputInfo") << " failed to retrieve data required by MET Task:  genMetTrue";
    }
  }
  if (!isMiniAODMET) {
    edm::Handle<GenMETCollection> genCalo;
    iEvent.getByToken(genMETsCaloToken_, genCalo);
    if (genCalo.isValid()) {
      const GenMETCollection *genmetcol = genCalo.product();
      const GenMET *genMetCalo = &(genmetcol->front());
      const double genMET = genMetCalo->pt();
      const double genMETPhi = genMetCalo->phi();

      mMETDifference_GenMETCalo->Fill(MET - genMET);
      mMETDeltaPhi_GenMETCalo->Fill(TMath::ACos(TMath::Cos(METPhi - genMETPhi)));
    } else {
      edm::LogInfo("OutputInfo") << " failed to retrieve data required by MET Task:  genMetCalo";
    }
  }
  if (isCaloMET) {
    const reco::CaloMET *calomet = &(caloMETs->front());
    // ==========================================================
    // Reconstructed MET Information
    const double caloMaxEtInEMTowers = calomet->maxEtInEmTowers();
    const double caloMaxEtInHadTowers = calomet->maxEtInHadTowers();
    const double caloEtFractionHadronic = calomet->etFractionHadronic();
    const double caloEmEtFraction = calomet->emEtFraction();
    const double caloHadEtInHB = calomet->hadEtInHB();
    const double caloHadEtInHO = calomet->hadEtInHO();
    const double caloHadEtInHE = calomet->hadEtInHE();
    const double caloHadEtInHF = calomet->hadEtInHF();
    const double caloEmEtInEB = calomet->emEtInEB();
    const double caloEmEtInEE = calomet->emEtInEE();
    const double caloEmEtInHF = calomet->emEtInHF();
    const double caloSETInpHF = calomet->CaloSETInpHF();
    const double caloSETInmHF = calomet->CaloSETInmHF();

    mCaloMaxEtInEmTowers->Fill(caloMaxEtInEMTowers);
    mCaloMaxEtInHadTowers->Fill(caloMaxEtInHadTowers);
    mCaloEtFractionHadronic->Fill(caloEtFractionHadronic);
    mCaloEmEtFraction->Fill(caloEmEtFraction);
    mCaloHadEtInHB->Fill(caloHadEtInHB);
    mCaloHadEtInHO->Fill(caloHadEtInHO);
    mCaloHadEtInHE->Fill(caloHadEtInHE);
    mCaloHadEtInHF->Fill(caloHadEtInHF);
    mCaloEmEtInEB->Fill(caloEmEtInEB);
    mCaloEmEtInEE->Fill(caloEmEtInEE);
    mCaloEmEtInHF->Fill(caloEmEtInHF);
    mCaloSETInpHF->Fill(caloSETInpHF);
    mCaloSETInmHF->Fill(caloSETInmHF);
  }
  if (isGenMET) {
    const GenMET *genmet;
    // Get Generated MET
    genmet = &(genMETs->front());

    const double NeutralEMEtFraction = genmet->NeutralEMEtFraction();
    const double NeutralHadEtFraction = genmet->NeutralHadEtFraction();
    const double ChargedEMEtFraction = genmet->ChargedEMEtFraction();
    const double ChargedHadEtFraction = genmet->ChargedHadEtFraction();
    const double MuonEtFraction = genmet->MuonEtFraction();
    const double InvisibleEtFraction = genmet->InvisibleEtFraction();

    mNeutralEMEtFraction->Fill(NeutralEMEtFraction);
    mNeutralHadEtFraction->Fill(NeutralHadEtFraction);
    mChargedEMEtFraction->Fill(ChargedEMEtFraction);
    mChargedHadEtFraction->Fill(ChargedHadEtFraction);
    mMuonEtFraction->Fill(MuonEtFraction);
    mInvisibleEtFraction->Fill(InvisibleEtFraction);
  }
  if (isPFMET) {
    const reco::PFMET *pfmet = &(pfMETs->front());
    mPFphotonEtFraction->Fill(pfmet->photonEtFraction());
    mPFphotonEt->Fill(pfmet->photonEt());
    mPFneutralHadronEtFraction->Fill(pfmet->neutralHadronEtFraction());
    mPFneutralHadronEt->Fill(pfmet->neutralHadronEt());
    mPFelectronEtFraction->Fill(pfmet->electronEtFraction());
    mPFelectronEt->Fill(pfmet->electronEt());
    mPFchargedHadronEtFraction->Fill(pfmet->chargedHadronEtFraction());
    mPFchargedHadronEt->Fill(pfmet->chargedHadronEt());
    mPFmuonEtFraction->Fill(pfmet->muonEtFraction());
    mPFmuonEt->Fill(pfmet->muonEt());
    mPFHFHadronEtFraction->Fill(pfmet->HFHadronEtFraction());
    mPFHFHadronEt->Fill(pfmet->HFHadronEt());
    mPFHFEMEtFraction->Fill(pfmet->HFEMEtFraction());
    mPFHFEMEt->Fill(pfmet->HFEMEt());
    // Reconstructed MET Information
  }
  if (isMiniAODMET) {
    const pat::MET *patmet = &(patMET->front());
    mMETUnc_JetResUp->Fill(MET - patmet->shiftedPt(pat::MET::JetResUp));
    mMETUnc_JetResDown->Fill(MET - patmet->shiftedPt(pat::MET::JetResDown));
    mMETUnc_JetEnUp->Fill(MET - patmet->shiftedPt(pat::MET::JetEnUp));
    mMETUnc_JetEnDown->Fill(MET - patmet->shiftedPt(pat::MET::JetEnDown));
    mMETUnc_MuonEnUp->Fill(MET - patmet->shiftedPt(pat::MET::MuonEnUp));
    mMETUnc_MuonEnDown->Fill(MET - patmet->shiftedPt(pat::MET::MuonEnDown));
    mMETUnc_ElectronEnUp->Fill(MET - patmet->shiftedPt(pat::MET::ElectronEnUp));
    mMETUnc_ElectronEnDown->Fill(MET - patmet->shiftedPt(pat::MET::ElectronEnDown));
    mMETUnc_TauEnUp->Fill(MET - patmet->shiftedPt(pat::MET::TauEnUp));
    mMETUnc_TauEnDown->Fill(MET - patmet->shiftedPt(pat::MET::TauEnDown));
    mMETUnc_UnclusteredEnUp->Fill(MET - patmet->shiftedPt(pat::MET::UnclusteredEnUp));
    mMETUnc_UnclusteredEnDown->Fill(MET - patmet->shiftedPt(pat::MET::UnclusteredEnDown));
    mMETUnc_PhotonEnUp->Fill(MET - patmet->shiftedPt(pat::MET::PhotonEnUp));
    mMETUnc_PhotonEnDown->Fill(MET - patmet->shiftedPt(pat::MET::PhotonEnDown));

    if (patmet->isPFMET()) {
      mPFphotonEtFraction->Fill(patmet->NeutralEMFraction());
      mPFneutralHadronEtFraction->Fill(patmet->NeutralHadEtFraction());
      mPFelectronEtFraction->Fill(patmet->ChargedEMEtFraction());
      mPFchargedHadronEtFraction->Fill(patmet->ChargedHadEtFraction());
      mPFmuonEtFraction->Fill(patmet->MuonEtFraction());
      mPFHFHadronEtFraction->Fill(patmet->Type6EtFraction());  // HFHadrons
      mPFHFEMEtFraction->Fill(patmet->Type7EtFraction());      // HFEMEt
    }
  }
}
