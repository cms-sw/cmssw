#include "METTester.h"
#include <format>

using namespace reco;
using namespace std;
using namespace edm;

METTester::METTester(const edm::ParameterSet &iConfig) {
  METType_ = iConfig.getUntrackedParameter<std::string>("METType");
  isCaloMET = std::string("calo") == METType_;
  isPFMET = std::string("pf") == METType_;
  isMHT = std::string("mht") == METType_;
  if (isMHT) {
	assert(mGenMetTrueLabel == mGenMetCaloLabel);
  }
  isGenMET = std::string("gen") == METType_;
  isMiniAODMET = std::string("miniaod") == METType_;

  inputMETLabel_ = iConfig.getParameter<std::string>("inputMETLabel");
  inputMHTLabel_ = iConfig.getParameter<std::string>("inputMHTLabel");
  runDir = iConfig.getUntrackedParameter<std::string>("runDir");

  mGenMetTrueLabel = iConfig.getParameter<std::string>("genMetTrueLabel");
  mGenMetCaloLabel = iConfig.getParameter<std::string>("genMetCaloLabel");
  
  if (isCaloMET) {
    caloMetToken_ = consumes<reco::CaloMETCollection>(inputMETLabel_);
	if (inputMHTLabel_ != "")
	  recoMhtToken_ = consumes<reco::METCollection>(inputMHTLabel_);
  }
  else if (isPFMET) {
	pfMetToken_ = consumes<reco::PFMETCollection>(inputMETLabel_);
	if (inputMHTLabel_ != "")
	  recoMhtToken_ = consumes<reco::METCollection>(inputMHTLabel_);
  }
  else if (isMHT) {
	recoMhtToken_ = consumes<reco::METCollection>(inputMHTLabel_);
	genMhtToken_ = consumes<reco::METCollection>(mGenMetTrueLabel);
	if (inputMETLabel_ != "")
	  pfMetToken_ = consumes<reco::PFMETCollection>(inputMETLabel_);
  }
  else if (isMiniAODMET) {
    patMetToken_ = consumes<pat::METCollection>(inputMETLabel_);
	if (inputMHTLabel_ != "")
	  recoMhtToken_ = consumes<reco::METCollection>(inputMHTLabel_);
  }
  else if (isGenMET) {
    genMetToken_ = consumes<reco::GenMETCollection>(inputMETLabel_);
	if (inputMHTLabel_ != "")
	  genMhtToken_ = consumes<reco::METCollection>(inputMHTLabel_);
  }

  if (!isMiniAODMET and !isMHT) {
	genMetTrueToken_ = consumes<reco::GenMETCollection>(mGenMetTrueLabel);
	genMetCaloToken_ = consumes<reco::GenMETCollection>(mGenMetCaloLabel);
	if (inputMHTLabel_ != "")
	  genMhtToken_ = consumes<reco::METCollection>(inputMHTLabel_);
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
  mGenMETTrue = nullptr;
  mGenMETCalo = nullptr;
  mMET = nullptr;
  mMHT = nullptr;
  mMET_vs_MHT = nullptr;
  mMET_Nvtx = nullptr;
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

void METTester::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const &iRun, edm::EventSetup const & /* iSetup */) {
  ibooker.setCurrentFolder(runDir + inputMETLabel_);

  mNvertex = ibooker.book1D("Nvertex", "Nvertex", 450, 0, 450);
  mMEx = ibooker.book1D("MEx", "MEx", 160, -metDiffEdge, metDiffEdge);
  mMEy = ibooker.book1D("MEy", "MEy", 160, -metDiffEdge, metDiffEdge);
  mMETSignPseudo = ibooker.book1D("METSignPseudo", "METSignPseudo", 25, 0, 24.5);
  mMETSignReal = ibooker.book1D("METSignReal", "METSignReal", 25, 0, 24.5);
  mGenMETTrue = ibooker.book1D("METGenTrue", "MET Gen True", 100, 0, metEdge);
  mGenMETCalo = ibooker.book1D("METGenTrue", "MET Gen True", 100, 0, metEdge);
  mMET = ibooker.book1D("MET", "MET (20 GeV binning)", 100, 0, metEdge);
  mMHT = ibooker.book1D("MHT", "MHT (20 GeV binning)", 100, 0, metEdge);
  mMET_Nvtx = ibooker.bookProfile("MET_Nvtx", "MET vs. nvtx", 450, 0., 450., 0., metEdge, "");
  mMETPhi = ibooker.book1D("METPhi", "METPhi", 100, -phiEdge, phiEdge);
  mSumET = ibooker.book1D("SumET", "SumET", 200, 0, 5000);
  mMETDiff_GenMETTrue = ibooker.book1D("METDiff_GenMETTrue", "METDiff_GenMETTrue", metDiffEdge, -metDiffEdge, metDiffEdge);
  mMETRatio_GenMETTrue = ibooker.book1D("METRatio_GenMETTrue", "METRatio_GenMETTrue", metRatioEdge, -metRatioEdge, metRatioEdge);
  mMETDeltaPhi_GenMETTrue = ibooker.book1D("METDeltaPhi_GenMETTrue", "METDeltaPhi_GenMETTrue", phiNbins, 0, phiEdge);

  mMET_vs_MHT = ibooker.book2D("METvsMHT", "MET vs MHT", 100, 0., metEdge, 100, 0., metEdge);
  mGenMETTrue_vs_GenMHT = ibooker.book2D("GenMETTruevsGenMHT", "Gen MET True vs Gen MHT", 100, 0., metEdge, 100, 0., metEdge);
  
  mGenMETTrue_vs_MET = ibooker.book2D("GenMETTruevsMET", "Gen MET True vs MET", 100, 0., metEdge, 100, 0., metEdge);
  mGenMETPhi_vs_MET = ibooker.book2D("GenMETPhivsMET", "Gen MET Phi vs MET", 100, -phiEdge, phiEdge, 100, 0., metEdge);
  mGenMETTrue_vs_mGenMETPhi = ibooker.book2D("GenMETTruevsGenMETPhi", "Gen MET True vs Gen MET Phi", 100, 0., metEdge, 100, -phiEdge, phiEdge);

  if (!isMHT) {
	mGenMETCalo_vs_MET = ibooker.book2D("GenMETCalovsMET", "Gen MET Calo vs MET", 100, 0., metEdge, 100, 0., metEdge);
	mGenMETCalo_vs_mGenMETPhi = ibooker.book2D("GenMETCalovsGenMETPhi", "Gen MET Calo vs Gen MET Phi", 100, 0., metEdge, 100, -phiEdge, phiEdge);
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
  if (!isMiniAODMET and !isMHT) {
    mMETDiff_GenMETCalo = ibooker.book1D("METDiff_GenMETCalo", "METDiff_GenMETCalo", metDiffEdge, -metDiffEdge, metDiffEdge);
    mMETRatio_GenMETCalo = ibooker.book1D("METRatio_GenMETCalo", "METRatio_GenMETCalo", metDiffEdge, -metDiffEdge, metDiffEdge);
    mMETDeltaPhi_GenMETCalo = ibooker.book1D("METDeltaPhi_GenMETCalo", "METDeltaPhi_GenMETCalo", 80, 0, phiEdge);
  }
  if (!isGenMET) {
	mMETDiff_vs_GenMETTrue = ibooker.book2D("METDiffvsGenMETTrue", "MET Diff vs Gen MET True",
											metDiffNbins, -metDiffEdge, metDiffEdge, metNbins, 0., metEdge);
	mMETDiff_vs_GenMETPhi = ibooker.book2D("METDiffvsGenMETPhi", "MET Diff vs Gen MET Phi",
										   metDiffNbins, -metDiffEdge, metDiffEdge, phiNbins, -phiEdge, phiEdge);
	mMETRatio_vs_GenMETTrue = ibooker.book2D("METRatiovsGenMETTrue", "MET Ratio vs Gen MET True",
											 metRatioNbins, -metRatioEdge, metRatioEdge, metNbins, 0., metEdge);
	mMETRatio_vs_GenMETPhi = ibooker.book2D("METRatiovsGenMETPhi", "MET Ratio vs Gen MET Phi",
											 metRatioNbins, -metRatioEdge, metRatioEdge, phiNbins, -phiEdge, phiEdge);
	mMETDeltaPhi_vs_GenMETTrue = ibooker.book2D("METDeltaPhivsGenMETTrue", "MET DeltaPhi vs Gen MET True",
												phiNbins, 0., phiEdge, metNbins, 0., metEdge);
	mMETDeltaPhi_vs_GenMETPhi = ibooker.book2D("METDeltaPhivsGenMETPhi", "MET DeltaPhi vs Gen MET Phi",
												phiNbins, 0., phiEdge, metNbins, 0., metEdge);
  }
  if (isCaloMET) {
	mCaloMaxEtInEmTowers = ibooker.book1D("CaloMaxEtInEmTowers", "CaloMaxEtInEmTowers", 300, 0, 1500);
	mCaloMaxEtInHadTowers = ibooker.book1D("CaloMaxEtInHadTowers", "CaloMaxEtInHadTowers", 300, 0, 1500);
	mCaloEtFractionHadronic = ibooker.book1D("CaloEtFractionHadronic", "CaloEtFractionHadronic", 100, 0, 1);
	mCaloEmEtFraction = ibooker.book1D("CaloEmEtFraction", "CaloEmEtFraction", 100, 0, 1);
	mCaloHadEtInHB = ibooker.book1D("CaloHadEtInHB", "CaloHadEtInHB", 200, 0, 2000);
	mCaloHadEtInHE = ibooker.book1D("CaloHadEtInHE", "CaloHadEtInHE", 100, 0, 500); 
	mCaloHadEtInHO = ibooker.book1D("CaloHadEtInHO", "CaloHadEtInHO", 100, 0, 200); 
	mCaloHadEtInHF = ibooker.book1D("CaloHadEtInHF", "CaloHadEtInHF", 100, 0, 200); 
	mCaloEmEtInEE = ibooker.book1D("CaloEmEtInEE", "CaloEmEtInEE", 100, 0, 500);
	mCaloEmEtInEB = ibooker.book1D("CaloEmEtInEB", "CaloEmEtInEB", 100, 0, 500);
	mCaloEmEtInHF = ibooker.book1D("CaloEmEtInHF", "CaloEmEtInHF", 100, 0, 500);
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
    edm::LogWarning("MissingInput") << __FUNCTION__ << ":" << __LINE__ << ": pvHandle handle with tag " << pvTokenTag_
                                    << " not found!";
    return;
  }
  const int nvtx = pvHandle->size();
  mNvertex->Fill(nvtx);

  if (isCaloMET) {
    iEvent.getByToken(caloMetToken_, caloMetHandle_);
    if (!caloMetHandle_.isValid()) return;
	if (inputMHTLabel_ != "") {
	  iEvent.getByToken(recoMhtToken_, recoMhtHandle_);
	  if (!recoMhtHandle_.isValid()) return;
	}
  } else if (isPFMET) {
    iEvent.getByToken(pfMetToken_, pfMetHandle_);
    if (!pfMetHandle_.isValid()) return;
	if (inputMHTLabel_ != "") {
	  iEvent.getByToken(recoMhtToken_, recoMhtHandle_);
	  if (!recoMhtHandle_.isValid()) return;
	}	
  } else if (isGenMET) {
    iEvent.getByToken(genMetToken_, genMetHandle_);
    if (!genMetHandle_.isValid()) return;
	if (inputMHTLabel_ != "") {
	  iEvent.getByToken(genMhtToken_, genMhtHandle_);
	  if (!genMhtHandle_.isValid()) return;
	}
  } else if (isMiniAODMET) {
    iEvent.getByToken(patMetToken_, patMetHandle_);
    if (!patMetHandle_.isValid()) return;
	if (inputMHTLabel_ != "") {
	  iEvent.getByToken(recoMhtToken_, recoMhtHandle_);
	  if (!recoMhtHandle_.isValid()) return;
	}
  } else if (isMHT) {
    iEvent.getByToken(recoMhtToken_, recoMhtHandle_);
    if (!recoMhtHandle_.isValid()) return;
	if (inputMETLabel_ != "") {
	  iEvent.getByToken(pfMetToken_, pfMetHandle_);
	  if (!pfMetHandle_.isValid()) return;
	}
  }

  // Two METs needed for 2D comparisons, eg. MET vs MHT
  reco::MET met, mht;
  bool fillMet2 = false;
  if (isCaloMET) {
    met = caloMetHandle_->front();
	if (inputMHTLabel_ != "") {
	  mht = recoMhtHandle_->front();
	  fillMet2 = true;
	}
  }
  else if (isPFMET) {
    met = pfMetHandle_->front();
  	if (inputMHTLabel_ != "") {
	  mht = recoMhtHandle_->front();
	  fillMet2 = true;
	}
  }
  else if (isGenMET) {
    met = genMetHandle_->front();
	if (inputMHTLabel_ != "") {
	  mht = genMhtHandle_->front();
	  fillMet2 = true;
	}
  }
  else if (isMiniAODMET) {
    met = patMetHandle_->front();
	if (inputMHTLabel_ != "") {
	  mht = recoMhtHandle_->front();
	  fillMet2 = true;
	}
  }
  else if (isMHT) {
	mht = recoMhtHandle_->front();
	if (inputMETLabel_ != "") {
	  met = pfMetHandle_->front();
	  fillMet2 = true;
	}
  }

  reco::MET& theMETobj = isMHT ? mht : met;
  
  const double SumET = theMETobj.sumEt();
  const double METSignPseudo = theMETobj.mEtSig();
  const double METSignReal = theMETobj.significance();  // covariance matrix to be fixed by JetMET

  double MET=-1., MHT=-1.;
  const double theMET = theMETobj.pt();
  if (isMHT) {
	MHT = mht.pt();
  }
  else {
	MET = met.pt();
  }
  const double MEx = theMETobj.px();
  const double MEy = theMETobj.py();
  const double METPhi = theMETobj.phi();

  mSumET->Fill(SumET);
  mMETSignPseudo->Fill(METSignPseudo);
  mMETSignReal->Fill(METSignReal);
  mMET->Fill(MET);
  if (fillMet2) {
	mMHT->Fill(MHT);
	mMET_vs_MHT->Fill(MET, MHT);
  }
  mMET_Nvtx->Fill((double)nvtx, MET);
  mMEx->Fill(MEx);
  mMEy->Fill(MEy);
  mMETPhi->Fill(METPhi);

  // Generated MET
  const reco::GenMET *genMetTrue = nullptr;
  // Get Generated MET for Resolution plots
  if (isMHT) {
    iEvent.getByToken(genMhtToken_, genMhtHandle_);
    if (!genMhtHandle_.isValid()) {
      return;
	}
  }
  else if (!isMiniAODMET) {
    edm::Handle<GenMETCollection> genTrue;
    iEvent.getByToken(genMetTrueToken_, genTrue);
    if (genTrue.isValid()) {
      const GenMETCollection *genmetcol = genTrue.product();
      genMetTrue = &(genmetcol->front());
    }
	else {
	  return;
	}
  }
  else {
    genMetTrue = patMetHandle_->front().genMET();
  }
  
  double genMET    = isMHT ? genMhtHandle_->at(0).pt()  : genMetTrue->pt();
  double genMETPhi = isMHT ? genMhtHandle_->at(0).phi() : genMetTrue->phi();
  double metDiff = theMET - genMET;
  double metRatio = theMET / genMET;
  double metDeltaPhi = TVector2::Phi_mpi_pi(METPhi - genMETPhi);

  mGenMETTrue->Fill(genMET);
  if (isMHT) {
	mGenMETTrue_vs_GenMHT->Fill(genMetTrue->pt(), genMET);
  }
  mMETDiff_GenMETTrue->Fill(metDiff);
  mMETRatio_GenMETTrue->Fill(metRatio);
  mMETDeltaPhi_GenMETTrue->Fill(metDeltaPhi);

  mGenMETTrue_vs_MET->Fill(genMET, theMET);
  mGenMETPhi_vs_MET->Fill(genMETPhi, theMET);
  mGenMETTrue_vs_mGenMETPhi->Fill(genMET, genMETPhi);

  // MET differences (Reco - Gen)
  if (!isGenMET) {
	std::cout << "=========================== " << metDiff << ", " << genMET << std::endl;
	mMETDiff_vs_GenMETTrue->Fill(metDiff, genMET);
	mMETDiff_vs_GenMETPhi->Fill(metDiff, genMETPhi);
	mMETRatio_vs_GenMETTrue->Fill(metRatio, genMET);
	mMETRatio_vs_GenMETPhi->Fill(metRatio, genMETPhi);
	mMETDeltaPhi_vs_GenMETTrue->Fill(metDeltaPhi, genMET);
	mMETDeltaPhi_vs_GenMETPhi->Fill(metDeltaPhi, genMETPhi);
  }
  
  if (!isMiniAODMET and !isMHT) {
    edm::Handle<GenMETCollection> genCalo;
    iEvent.getByToken(genMetCaloToken_, genCalo);
    if (genCalo.isValid()) {
      const GenMETCollection *genmetcol = genCalo.product();
      const GenMET *genMetCalo = &(genmetcol->front());
      const double genMET = genMetCalo->pt();
      const double genMETPhi = genMetCalo->phi();

	  mGenMETCalo->Fill(genMET);
      mMETDiff_GenMETCalo->Fill(theMET - genMET);
      mMETRatio_GenMETCalo->Fill(theMET / genMET);
      mMETDeltaPhi_GenMETCalo->Fill(TVector2::Phi_mpi_pi(METPhi - genMETPhi));

	  mGenMETCalo_vs_MET->Fill(genMET, theMET);
	  mGenMETCalo_vs_mGenMETPhi->Fill(genMET, genMETPhi);
    }
	else {
      edm::LogInfo("OutputInfo") << " failed to retrieve data required by MET Task: genMetCalo";
    }
  }
  if (isCaloMET) {
    const reco::CaloMET *calomet = &(caloMetHandle_->front());
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
  }
  if (isGenMET) {
    const GenMET *genmet;
    // Get Generated MET
    genmet = &(genMetHandle_->front());

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
    const reco::PFMET *pfmet = &(pfMetHandle_->front());
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
    const pat::MET *patmet = &(patMetHandle_->front());
    mMETUnc_JetResUp->Fill(theMET - patmet->shiftedPt(pat::MET::JetResUp));
    mMETUnc_JetResDown->Fill(theMET - patmet->shiftedPt(pat::MET::JetResDown));
    mMETUnc_JetEnUp->Fill(theMET - patmet->shiftedPt(pat::MET::JetEnUp));
    mMETUnc_JetEnDown->Fill(theMET - patmet->shiftedPt(pat::MET::JetEnDown));
    mMETUnc_MuonEnUp->Fill(theMET - patmet->shiftedPt(pat::MET::MuonEnUp));
    mMETUnc_MuonEnDown->Fill(theMET - patmet->shiftedPt(pat::MET::MuonEnDown));
    mMETUnc_ElectronEnUp->Fill(theMET - patmet->shiftedPt(pat::MET::ElectronEnUp));
    mMETUnc_ElectronEnDown->Fill(theMET - patmet->shiftedPt(pat::MET::ElectronEnDown));
    mMETUnc_TauEnUp->Fill(theMET - patmet->shiftedPt(pat::MET::TauEnUp));
    mMETUnc_TauEnDown->Fill(theMET - patmet->shiftedPt(pat::MET::TauEnDown));
    mMETUnc_UnclusteredEnUp->Fill(theMET - patmet->shiftedPt(pat::MET::UnclusteredEnUp));
    mMETUnc_UnclusteredEnDown->Fill(theMET - patmet->shiftedPt(pat::MET::UnclusteredEnDown));
    mMETUnc_PhotonEnUp->Fill(theMET - patmet->shiftedPt(pat::MET::PhotonEnUp));
    mMETUnc_PhotonEnDown->Fill(theMET - patmet->shiftedPt(pat::MET::PhotonEnDown));

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
  desc.add<std::string>("inputMETLabel", "pfMet");
  desc.add<std::string>("inputMHTLabel", "");
  desc.addUntracked<std::string>("METType", "pf");
  desc.add<std::string>("genMetTrueLabel", "genMetTrue");
  desc.add<std::string>("genMetCaloLabel", "genMetCalo");
  descriptions.addWithDefaultLabel(desc);
}
