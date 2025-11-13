#include "METTester.h"
#include <format>

using namespace reco;
using namespace std;
using namespace edm;

METTester::METTester(const edm::ParameterSet &iConfig) {
  inputMETLabel_ = iConfig.getParameter<edm::InputTag>("inputMETLabel");
  runDir = iConfig.getUntrackedParameter<std::string>("runDir");

  METType_ = iConfig.getUntrackedParameter<std::string>("METType");
  isCaloMET = std::string("calo") == METType_;
  isPFMET = std::string("pf") == METType_;
  isGenMET = std::string("gen") == METType_;
  isMiniAODMET = std::string("miniaod") == METType_;

  if (isCaloMET)
    caloMETsToken_ = consumes<reco::CaloMETCollection>(inputMETLabel_);
  else if (isPFMET)
    pfMETsToken_ = consumes<reco::PFMETCollection>(inputMETLabel_);
  else if (isMiniAODMET)
    patMETToken_ = consumes<pat::METCollection>(inputMETLabel_);
  else if (isGenMET)
    genMETsToken_ = consumes<reco::GenMETCollection>(inputMETLabel_);

  if (!isMiniAODMET) {
    genMETsTrueToken_ = consumes<reco::GenMETCollection>(edm::InputTag("genMetTrue"));
    genMETsCaloToken_ = consumes<reco::GenMETCollection>(edm::InputTag("genMetCalo"));
  }

  pvTokenTag_ = iConfig.getParameter<edm::InputTag>("primaryVertices");
  pvToken_ = consumes<std::vector<reco::Vertex>>(pvTokenTag_);

  // Events variables
  mNvertex = nullptr;

  // Common variables
  mMEx = nullptr;
  mMEy = nullptr;
  mMETSignPseudo = nullptr;
  mMETSignReal = nullptr;
  mMET = nullptr;
  mMETFine = nullptr;
  mMET_Nvtx = nullptr;
  mMETEta = nullptr;
  mMETPhi = nullptr;
  mSumET = nullptr;

  mMETDiff_GenMETTrue = nullptr;
  mMETRatio_GenMETTrue = nullptr;
  mMETDeltaPhi_GenMETTrue = nullptr;

  mMETDiff_GenMETCalo = nullptr;
  mMETRatio_GenMETCalo = nullptr;
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
}

std::string METTester::binStr(float left, float right, bool roundInt) {
  std::string out;
  if (roundInt) {
    out = std::to_string((int)left) + "to" + std::to_string((int)right);
  } else {
    out = std::format("{:.2f}", left) + "to" + std::format("{:.2f}", right);
    std::replace(out.begin(), out.end(), '.', 'p');
  }
  return out;
}

void METTester::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const &iRun, edm::EventSetup const & /* iSetup */) {
  ibooker.setCurrentFolder(runDir + inputMETLabel_.label());

  mNvertex = ibooker.book1D("Nvertex", "Nvertex", 450, 0, 450);
  mMEx = ibooker.book1D("MEx", "MEx", 160, -800, 800);
  mMEy = ibooker.book1D("MEy", "MEy", 160, -800, 800);
  mMETSignPseudo = ibooker.book1D("METSignPseudo", "METSignPseudo", 25, 0, 24.5);
  mMETSignReal = ibooker.book1D("METSignReal", "METSignReal", 25, 0, 24.5);
  mMET = ibooker.book1D("MET", "MET (20 GeV binning)", 100, 0, 2000);
  mMETFine = ibooker.book1D("METFine", "MET (2 GeV binning)", 1000, 0, 2000);
  mMET_Nvtx = ibooker.bookProfile("MET_Nvtx", "MET vs. nvtx", 450, 0., 450., 0., 2000., "");
  mMETEta = ibooker.book1D("METEta", "METEta", 80, -6, 6);
  mMETPhi = ibooker.book1D("METPhi", "METPhi", 80, -4, 4);
  mSumET = ibooker.book1D("SumET", "SumET", 200, 0, 5000);  // 10GeV
  mMETDiff_GenMETTrue = ibooker.book1D("METDiff_GenMETTrue", "METDiff_GenMETTrue", 800, -800, 800);
  mMETRatio_GenMETTrue = ibooker.book1D("METRatio_GenMETTrue", "METRatio_GenMETTrue", 800, -800, 800);
  mMETDeltaPhi_GenMETTrue = ibooker.book1D("METDeltaPhi_GenMETTrue", "METDeltaPhi_GenMETTrue", 80, 0, 4);

  for (unsigned metIdx = 0; metIdx < mNMETBins; ++metIdx) {
    std::string title = "MET_MET" + binStr(mMETBins[metIdx], mMETBins[metIdx + 1], true);
    mMET_METBins[metIdx] = ibooker.book1D(title.c_str(), title.c_str(), 50, mMETBins[metIdx], mMETBins[metIdx + 1]);
  }
  for (unsigned metIdx = 0; metIdx < mNPhiBins; ++metIdx) {
    std::string title = "MET_Phi" + binStr(mPhiBins[metIdx], mPhiBins[metIdx + 1], false);
    mMET_PhiBins[metIdx] = ibooker.book1D(title.c_str(), title.c_str(), 600, -600, 600);
  }

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
    mMETDiff_GenMETCalo = ibooker.book1D("METDiff_GenMETCalo", "METDiff_GenMETCalo", 600, -600, 600);
    mMETRatio_GenMETCalo = ibooker.book1D("METRatio_GenMETCalo", "METRatio_GenMETCalo", 600, -600, 600);
    mMETDeltaPhi_GenMETCalo = ibooker.book1D("METDeltaPhi_GenMETCalo", "METDeltaPhi_GenMETCalo", 80, 0, 4);
  }
  if (!isGenMET) {
    for (unsigned metIdx = 0; metIdx < mNMETBins; ++metIdx) {
      const std::string title = "_GenMETTrue_MET" + binStr(mMETBins[metIdx], mMETBins[metIdx + 1], true);
      mMETDiff_GenMETTrue_METBins[metIdx] =
          ibooker.book1D(("METDiff" + title).c_str(), ("METDiff" + title).c_str(), 600, -600, 600);
      mMETRatio_GenMETTrue_METBins[metIdx] =
          ibooker.book1D(("METRatio" + title).c_str(), ("METRatio" + title).c_str(), 600, -600, 600);
      mMETDeltaPhi_GenMETTrue_METBins[metIdx] =
          ibooker.book1D(("METDeltaPhi" + title).c_str(), ("METDeltaPhi" + title).c_str(), 80, 0, 4);
    }
    for (unsigned metIdx = 0; metIdx < mNPhiBins; ++metIdx) {
      const std::string title = "_GenMETTrue_Phi" + binStr(mPhiBins[metIdx], mPhiBins[metIdx + 1], false);
      mMETDiff_GenMETTrue_PhiBins[metIdx] =
          ibooker.book1D(("METDiff" + title).c_str(), ("METDiff" + title).c_str(), 600, -600, 600);
      mMETRatio_GenMETTrue_PhiBins[metIdx] =
          ibooker.book1D(("METRatio" + title).c_str(), ("METRatio" + title).c_str(), 600, -600, 600);
      mMETDeltaPhi_GenMETTrue_PhiBins[metIdx] =
          ibooker.book1D(("METDeltaPhi" + title).c_str(), ("METDeltaPhi" + title).c_str(), 80, 0, 4);
    }
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
      mPFphotonEt = ibooker.book1D("photonEt", "photonEt", 150, 0, 1500);
      mPFneutralHadronEt = ibooker.book1D("neutralHadronEt", "neutralHadronEt", 100, 0, 1000);
      mPFelectronEt = ibooker.book1D("electronEt", "electronEt", 100, 0, 1000);
      mPFchargedHadronEt = ibooker.book1D("chargedHadronEt", "chargedHadronEt", 150, 0, 1500);
      mPFmuonEt = ibooker.book1D("muonEt", "muonEt", 100, 0, 1000);
      mPFHFHadronEt = ibooker.book1D("HFHadronEt", "HFHadronEt", 100, 0, 300);
      mPFHFEMEt = ibooker.book1D("HFEMEt", "HFEMEt", 50, 0, 150);
    }
  }
}

void METTester::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  edm::Handle<reco::VertexCollection> pvHandle;
  iEvent.getByToken(pvToken_, pvHandle);
  if (!pvHandle.isValid()) {
    edm::LogWarning("MissingInput") << __FUNCTION__ << ":" << __LINE__ << ": pvHandle handle with tag " << pvTokenTag_ << " not found!";
    return;
  }
  const int nvtx = pvHandle->size();
  mNvertex->Fill(nvtx);

  edm::Handle<CaloMETCollection> caloMETs;
  edm::Handle<PFMETCollection> pfMETs;
  edm::Handle<GenMETCollection> genMETs;
  edm::Handle<pat::METCollection> patMET;

  if (isCaloMET) {
    iEvent.getByToken(caloMETsToken_, caloMETs);
    if (!caloMETs.isValid())
      return;
  } else if (isPFMET) {
    iEvent.getByToken(pfMETsToken_, pfMETs);
    if (!pfMETs.isValid())
      return;
  } else if (isGenMET) {
    iEvent.getByToken(genMETsToken_, genMETs);
    if (!genMETs.isValid())
      return;
  } else if (isMiniAODMET) {
    iEvent.getByToken(patMETToken_, patMET);
    if (!patMET.isValid())
      return;
  }

  reco::MET met;
  if (isCaloMET)
    met = caloMETs->front();
  else if (isPFMET)
    met = pfMETs->front();
  else if (isGenMET)
    met = genMETs->front();
  else if (isMiniAODMET)
    met = patMET->front();

  const double SumET = met.sumEt();
  const double METSignPseudo = met.mEtSig();
  const double METSignReal = met.significance();  // covariance matrix to be fixed by JetMET

  const double MET = met.pt();
  const double MEx = met.px();
  const double MEy = met.py();
  const double METEta = met.eta();
  const double METPhi = met.phi();

  mSumET->Fill(SumET);
  mMETSignPseudo->Fill(METSignPseudo);
  mMETSignReal->Fill(METSignReal);
  mMET->Fill(MET);
  mMETFine->Fill(MET);
  mMET_Nvtx->Fill((double)nvtx, MET);
  mMEx->Fill(MEx);
  mMEy->Fill(MEy);
  mMETEta->Fill(METEta);
  mMETPhi->Fill(METPhi);

  for (unsigned metIdx = 0; metIdx < mNMETBins; ++metIdx) {
    if (MET >= mMETBins[metIdx] && MET < mMETBins[metIdx + 1])
      mMET_METBins[metIdx]->Fill(MET);
  }
  for (unsigned metIdx = 0; metIdx < mNPhiBins; ++metIdx) {
    if (METPhi >= mPhiBins[metIdx] && METPhi < mPhiBins[metIdx + 1])
      mMET_PhiBins[metIdx]->Fill(MET);
  }

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
    double metDiff = MET - genMET;
    double metRatio = MET / genMET;
    double metDeltaPhi = TVector2::Phi_mpi_pi(METPhi - genMETPhi);

    mMETDiff_GenMETTrue->Fill(metDiff);
    mMETRatio_GenMETTrue->Fill(metRatio);
    mMETDeltaPhi_GenMETTrue->Fill(metDeltaPhi);

    if (!isGenMET) {
      // MET difference in MET bins
      for (unsigned metIdx = 0; metIdx < mNMETBins; ++metIdx) {
        if (MET >= mMETBins[metIdx] && MET < mMETBins[metIdx + 1]) {
          mMETDiff_GenMETTrue_METBins[metIdx]->Fill(metDiff);
          mMETRatio_GenMETTrue_METBins[metIdx]->Fill(metRatio);
          mMETDeltaPhi_GenMETTrue_METBins[metIdx]->Fill(metDeltaPhi);
        }
      }
      // MET difference in Phi bins
      for (unsigned metIdx = 0; metIdx < mNPhiBins; ++metIdx) {
        if (METPhi >= mPhiBins[metIdx] && METPhi < mPhiBins[metIdx + 1]) {
          mMETDiff_GenMETTrue_PhiBins[metIdx]->Fill(metDiff);
          mMETRatio_GenMETTrue_PhiBins[metIdx]->Fill(metRatio);
          mMETDeltaPhi_GenMETTrue_PhiBins[metIdx]->Fill(metDeltaPhi);
        }
      }
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

      mMETDiff_GenMETCalo->Fill(MET - genMET);
      mMETRatio_GenMETCalo->Fill(MET / genMET);
      mMETDeltaPhi_GenMETCalo->Fill(TVector2::Phi_mpi_pi(METPhi - genMETPhi));
    } else {
      edm::LogInfo("OutputInfo") << " failed to retrieve data required by MET Task: genMetCalo";
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

//------------------------------------------------------------------------------
// fill description
//------------------------------------------------------------------------------
void METTester::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  // Default MET validation offline
  desc.addUntracked<std::string>("runDir", "JetMET/METValidation/");
  desc.add<edm::InputTag>("primaryVertices", edm::InputTag("PixelVertices"));
  desc.add<edm::InputTag>("inputMETLabel", edm::InputTag("pfMet"));
  desc.addUntracked<std::string>("METType", "pf");
  desc.add<edm::InputTag>("genMetTrue", edm::InputTag("genMetTrue"));
  desc.add<edm::InputTag>("genMetCalo", edm::InputTag("genMetCalo"));
  descriptions.addWithDefaultLabel(desc);
}
