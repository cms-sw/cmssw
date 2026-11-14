#include "METTester.h"
#include "FWCore/Utilities/interface/Exception.h"

METTester::METTester(const edm::ParameterSet &iConfig) {
  METType_ = iConfig.getUntrackedParameter<std::string>("METType");
  isCaloMET = METType_ == "calo";
  isPFMET = METType_ == "pf";
  isMHT = METType_ == "mht";
  isGenMET = METType_ == "gen";
  isMiniAODMET = METType_ == "miniaod";
  if (!(isCaloMET or isPFMET or isMHT or isGenMET or isMiniAODMET)) {
    throw cms::Exception("Configuration")
        << "METTester: unknown METType '" << METType_ << "'. Use one of: calo, pf, mht, gen, miniaod.";
  }

  inputMETLabel_ = iConfig.getParameter<std::string>("inputMETLabel");
  inputMHTLabel_ = iConfig.getParameter<std::string>("inputMHTLabel");
  runDir = iConfig.getUntrackedParameter<std::string>("runDir");
  outFolder_ = iConfig.getUntrackedParameter<std::string>("outFolder");

  mGenMetTrueLabel = iConfig.getParameter<std::string>("genMetTrueLabel");
  mGenMetCaloLabel = iConfig.getParameter<std::string>("genMetCaloLabel");
  mGenMhtLabel = iConfig.getParameter<std::string>("genMhtLabel");

  primaryLabel_ = isMHT ? inputMHTLabel_ : inputMETLabel_;
  companionLabel_ = isMHT ? inputMETLabel_ : inputMHTLabel_;
  hasCompanion_ = !companionLabel_.empty();
  if (primaryLabel_.empty()) {
    throw cms::Exception("Configuration") << "METTester: METType '" << METType_ << "' requires a non-empty "
                                          << (isMHT ? "inputMHTLabel" : "inputMETLabel") << ".";
  }

  primaryToken_ = consumes<METView>(primaryLabel_);
  if (hasCompanion_)
    companionToken_ = consumes<METView>(companionLabel_);

  // gen MET true: reference for MET, and needed for the gen MET vs gen MHT comparison
  if (!isMiniAODMET or hasCompanion_)
    genMetTrueToken_ = consumes<METView>(mGenMetTrueLabel);
  if (!isMiniAODMET and !isMHT)
    genMetCaloToken_ = consumes<METView>(mGenMetCaloLabel);
  // gen MHT: reference for MHT, and needed for the gen MET vs gen MHT comparison
  if (isMHT or hasCompanion_)
    genMhtToken_ = consumes<METView>(mGenMhtLabel);

  pvTokenTag_ = iConfig.getParameter<edm::InputTag>("primaryVertices");
  pvToken_ = consumes<std::vector<reco::Vertex>>(pvTokenTag_);
}

void METTester::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const &iRun, edm::EventSetup const & /* iSetup */) {
  // output folder: `outFolder` if set, otherwise the module label
  ibooker.setCurrentFolder(runDir + (outFolder_.empty() ? moduleDescription().moduleLabel() : outFolder_));

  mNvertex = ibooker.book1D("Nvertex", "Nvertex", 450, 0, 450);
  mMEx = ibooker.book1D("MEx", "MEx", 160, -metDiffEdge, metDiffEdge);
  mMEy = ibooker.book1D("MEy", "MEy", 160, -metDiffEdge, metDiffEdge);
  mMETSignPseudo = ibooker.book1D("METSignPseudo", "METSignPseudo", 25, 0, 24.5);
  mMETSignReal = ibooker.book1D("METSignReal", "METSignReal", 25, 0, 24.5);
  mGenMETTrue = ibooker.book1D("GenMETTrue", "MET Gen True (20 GeV binning)", metCoarseNbins, 0, metEdge);
  mMET = ibooker.book1D("MET", "MET (20 GeV binning)", metCoarseNbins, 0, metEdge);
  mMET_Nvtx = ibooker.bookProfile("MET_Nvtx", "MET vs. nvtx", 450, 0., 450., 0., metEdge, "");
  mMETPhi = ibooker.book1D("METPhi", "METPhi", phiNbins, -phiEdge, phiEdge);
  mSumET = ibooker.book1D("SumET", "SumET", 200, 0, 5000);
  mMETDiff_GenMETTrue =
      ibooker.book1D("METDiff_GenMETTrue", "METDiff_GenMETTrue", metDiffNbins, -metDiffEdge, metDiffEdge);
  mMETRatio_GenMETTrue = ibooker.book1D("METRatio_GenMETTrue", "METRatio_GenMETTrue", metRatioNbins, 0., metRatioEdge);
  mMETDeltaPhi_GenMETTrue =
      ibooker.book1D("METDeltaPhi_GenMETTrue", "METDeltaPhi_GenMETTrue", phiNbins, -phiEdge, phiEdge);

  mGenMETTrue_vs_MET = ibooker.book2D(
      "GenMETTruevsMET", "Gen MET True vs MET", metCoarseNbins, 0., metEdge, metCoarseNbins, 0., metEdge);
  mGenMETPhi_vs_MET =
      ibooker.book2D("GenMETPhivsMET", "Gen MET Phi vs MET", phiNbins, -phiEdge, phiEdge, metNbins, 0., metEdge);
  mGenMETTrue_vs_mGenMETPhi = ibooker.book2D(
      "GenMETTruevsGenMETPhi", "Gen MET True vs Gen MET Phi", metNbins, 0., metEdge, phiNbins, -phiEdge, phiEdge);

  if (!isMiniAODMET and !isMHT) {
    mGenMETCalo = ibooker.book1D("GenMETCalo", "MET Gen Calo (20 GeV binning)", metCoarseNbins, 0, metEdge);
    mGenMETCalo_vs_MET = ibooker.book2D(
        "GenMETCalovsMET", "Gen MET Calo vs MET", metCoarseNbins, 0., metEdge, metCoarseNbins, 0., metEdge);
    mGenMETCalo_vs_mGenMETPhi = ibooker.book2D(
        "GenMETCalovsGenMETPhi", "Gen MET Calo vs Gen MET Phi", metNbins, 0., metEdge, phiNbins, -phiEdge, phiEdge);
    mMETDiff_GenMETCalo =
        ibooker.book1D("METDiff_GenMETCalo", "METDiff_GenMETCalo", metDiffNbins, -metDiffEdge, metDiffEdge);
    mMETRatio_GenMETCalo =
        ibooker.book1D("METRatio_GenMETCalo", "METRatio_GenMETCalo", metRatioNbins, 0., metRatioEdge);
    mMETDeltaPhi_GenMETCalo =
        ibooker.book1D("METDeltaPhi_GenMETCalo", "METDeltaPhi_GenMETCalo", phiNbins, -phiEdge, phiEdge);
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
    mMETUnc_PhotonEnUp = ibooker.book1D("METUnc_PhotonEnUp", "METUnc_PhotonEnUp", 200, -10, 10);
    mMETUnc_PhotonEnDown = ibooker.book1D("METUnc_PhotonEnDown", "METUnc_PhotonEnDown", 200, -10, 10);
  }

  if (!isGenMET) {
    mMETDiff_vs_GenMETTrue = ibooker.book2D("METDiffvsGenMETTrue",
                                            "MET Diff vs Gen MET True",
                                            metDiffNbins,
                                            -metDiffEdge,
                                            metDiffEdge,
                                            metNbins,
                                            0.,
                                            metEdge);
    mMETDiff_vs_GenMETPhi = ibooker.book2D("METDiffvsGenMETPhi",
                                           "MET Diff vs Gen MET Phi",
                                           metDiffNbins,
                                           -metDiffEdge,
                                           metDiffEdge,
                                           phiNbins,
                                           -phiEdge,
                                           phiEdge);
    mMETRatio_vs_GenMETTrue = ibooker.book2D(
        "METRatiovsGenMETTrue", "MET Ratio vs Gen MET True", metRatioNbins, 0., metRatioEdge, metNbins, 0., metEdge);
    mMETRatio_vs_GenMETPhi = ibooker.book2D(
        "METRatiovsGenMETPhi", "MET Ratio vs Gen MET Phi", metRatioNbins, 0., metRatioEdge, phiNbins, -phiEdge, phiEdge);
    mMETDeltaPhi_vs_GenMETTrue = ibooker.book2D(
        "METDeltaPhivsGenMETTrue", "MET DeltaPhi vs Gen MET True", phiNbins, -phiEdge, phiEdge, metNbins, 0., metEdge);
    mMETDeltaPhi_vs_GenMETPhi = ibooker.book2D("METDeltaPhivsGenMETPhi",
                                               "MET DeltaPhi vs Gen MET Phi",
                                               phiNbins,
                                               -phiEdge,
                                               phiEdge,
                                               phiNbins,
                                               -phiEdge,
                                               phiEdge);
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

  // MET vs MHT comparisons live in a dedicated folder, shared by MET and MHT
  if (hasCompanion_) {
    ibooker.setCurrentFolder(runDir + "METvsMHT/" + inputMETLabel_ + "_" + inputMHTLabel_);
    mMET_vs_MHT = ibooker.book2D("METvsMHT", "MET vs MHT", metCoarseNbins, 0., metEdge, metCoarseNbins, 0., metEdge);
    mGenMETTrue_vs_GenMHT = ibooker.book2D(
        "GenMETTruevsGenMHT", "Gen MET True vs Gen MHT", metCoarseNbins, 0., metEdge, metCoarseNbins, 0., metEdge);
  }
}

void METTester::warnOnce(const std::string &label, const std::string &msg) {
  if (warned_.insert(label).second) {
    edm::LogWarning("METTester") << "[" << primaryLabel_ << "] " << msg << " (reported only once)";
  }
}

const reco::MET *METTester::firstOrNull(const edm::Event &iEvent,
                                        const edm::EDGetTokenT<METView> &token,
                                        const std::string &label) {
  const auto &handle = iEvent.getHandle(token);
  if (!handle.isValid()) {
    warnOnce(label, "Collection '" + label + "' not found; dependent histograms are not filled.");
    return nullptr;
  }
  if (handle->empty()) {
    warnOnce(label + "_empty", "Collection '" + label + "' is empty; dependent histograms are not filled.");
    return nullptr;
  }
  return &handle->front();
}

void METTester::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  int nvtx = -1;
  const auto &pvHandle = iEvent.getHandle(pvToken_);
  if (pvHandle.isValid()) {
    nvtx = pvHandle->size();
    mNvertex->Fill(nvtx);
  } else {
    warnOnce(pvTokenTag_.encode(), "Vertex collection '" + pvTokenTag_.encode() + "' not found.");
  }

  const reco::MET *primary = firstOrNull(iEvent, primaryToken_, primaryLabel_);
  if (!primary)
    return;

  const double theMET = primary->pt();
  const double METPhi = primary->phi();

  mSumET->Fill(primary->sumEt());
  mMETSignPseudo->Fill(primary->mEtSig());
  mMETSignReal->Fill(primary->significance());  // covariance matrix to be fixed by JetMET
  mMET->Fill(theMET);
  if (nvtx >= 0)
    mMET_Nvtx->Fill(nvtx, theMET);
  mMEx->Fill(primary->px());
  mMEy->Fill(primary->py());
  mMETPhi->Fill(METPhi);

  // Generator-level reference: gen MHT for MHT, gen MET true otherwise
  const reco::MET *genRef = nullptr;
  if (isMHT) {
    genRef = firstOrNull(iEvent, genMhtToken_, mGenMhtLabel);
  } else if (isMiniAODMET) {
    const auto *patmet = dynamic_cast<const pat::MET *>(primary);
    genRef = patmet ? patmet->genMET() : nullptr;
    if (patmet and !genRef)
      warnOnce("patGenMET", "pat::MET '" + primaryLabel_ + "' has no associated gen MET.");
  } else {
    genRef = firstOrNull(iEvent, genMetTrueToken_, mGenMetTrueLabel);
  }
  if (genRef)
    fillGenComparison(*primary, *genRef);

  if (!isMiniAODMET and !isMHT) {
    if (const reco::MET *genCalo = firstOrNull(iEvent, genMetCaloToken_, mGenMetCaloLabel))
      fillGenCaloComparison(*primary, *genCalo);
  }

  if (hasCompanion_) {
    if (const reco::MET *companion = firstOrNull(iEvent, companionToken_, companionLabel_)) {
      const reco::MET &met = isMHT ? *companion : *primary;
      const reco::MET &mht = isMHT ? *primary : *companion;
      mMET_vs_MHT->Fill(met.pt(), mht.pt());
    }
    const reco::MET *genMet = firstOrNull(iEvent, genMetTrueToken_, mGenMetTrueLabel);
    const reco::MET *genMht = firstOrNull(iEvent, genMhtToken_, mGenMhtLabel);
    if (genMet and genMht)
      mGenMETTrue_vs_GenMHT->Fill(genMet->pt(), genMht->pt());
  }

  // Type-specific variables
  if (isCaloMET) {
    const auto *calomet = dynamic_cast<const reco::CaloMET *>(primary);
    if (!calomet)
      throw cms::Exception("Configuration") << "METTester: '" << primaryLabel_ << "' is not a reco::CaloMET.";
    fillCaloMET(*calomet);
  } else if (isPFMET) {
    const auto *pfmet = dynamic_cast<const reco::PFMET *>(primary);
    if (!pfmet)
      throw cms::Exception("Configuration") << "METTester: '" << primaryLabel_ << "' is not a reco::PFMET.";
    fillPFMET(*pfmet);
  } else if (isGenMET) {
    const auto *genmet = dynamic_cast<const reco::GenMET *>(primary);
    if (!genmet)
      throw cms::Exception("Configuration") << "METTester: '" << primaryLabel_ << "' is not a reco::GenMET.";
    fillGenMET(*genmet);
  } else if (isMiniAODMET) {
    const auto *patmet = dynamic_cast<const pat::MET *>(primary);
    if (!patmet)
      throw cms::Exception("Configuration") << "METTester: '" << primaryLabel_ << "' is not a pat::MET.";
    fillPatMET(*patmet, theMET);
  }
}

void METTester::fillGenComparison(const reco::MET &reco, const reco::MET &gen) {
  const double theMET = reco.pt();
  const double genMET = gen.pt();
  const double genMETPhi = gen.phi();
  const double metDiff = theMET - genMET;
  const double metDeltaPhi = TVector2::Phi_mpi_pi(reco.phi() - genMETPhi);
  const bool validRatio = genMET > mEpsilon;
  const double metRatio = validRatio ? theMET / genMET : 0.;

  mGenMETTrue->Fill(genMET);
  mMETDiff_GenMETTrue->Fill(metDiff);
  if (validRatio)
    mMETRatio_GenMETTrue->Fill(metRatio);
  mMETDeltaPhi_GenMETTrue->Fill(metDeltaPhi);

  mGenMETTrue_vs_MET->Fill(genMET, theMET);
  mGenMETPhi_vs_MET->Fill(genMETPhi, theMET);
  mGenMETTrue_vs_mGenMETPhi->Fill(genMET, genMETPhi);

  // MET differences (Reco - Gen)
  if (!isGenMET) {
    mMETDiff_vs_GenMETTrue->Fill(metDiff, genMET);
    mMETDiff_vs_GenMETPhi->Fill(metDiff, genMETPhi);
    if (validRatio) {
      mMETRatio_vs_GenMETTrue->Fill(metRatio, genMET);
      mMETRatio_vs_GenMETPhi->Fill(metRatio, genMETPhi);
    }
    mMETDeltaPhi_vs_GenMETTrue->Fill(metDeltaPhi, genMET);
    mMETDeltaPhi_vs_GenMETPhi->Fill(metDeltaPhi, genMETPhi);
  }
}

void METTester::fillGenCaloComparison(const reco::MET &reco, const reco::MET &genCalo) {
  const double theMET = reco.pt();
  const double genMET = genCalo.pt();
  const double genMETPhi = genCalo.phi();

  mGenMETCalo->Fill(genMET);
  mMETDiff_GenMETCalo->Fill(theMET - genMET);
  if (genMET > mEpsilon)
    mMETRatio_GenMETCalo->Fill(theMET / genMET);
  mMETDeltaPhi_GenMETCalo->Fill(TVector2::Phi_mpi_pi(reco.phi() - genMETPhi));

  mGenMETCalo_vs_MET->Fill(genMET, theMET);
  mGenMETCalo_vs_mGenMETPhi->Fill(genMET, genMETPhi);
}

void METTester::fillCaloMET(const reco::CaloMET &calomet) {
  mCaloMaxEtInEmTowers->Fill(calomet.maxEtInEmTowers());
  mCaloMaxEtInHadTowers->Fill(calomet.maxEtInHadTowers());
  mCaloEtFractionHadronic->Fill(calomet.etFractionHadronic());
  mCaloEmEtFraction->Fill(calomet.emEtFraction());
  mCaloHadEtInHB->Fill(calomet.hadEtInHB());
  mCaloHadEtInHO->Fill(calomet.hadEtInHO());
  mCaloHadEtInHE->Fill(calomet.hadEtInHE());
  mCaloHadEtInHF->Fill(calomet.hadEtInHF());
  mCaloEmEtInEB->Fill(calomet.emEtInEB());
  mCaloEmEtInEE->Fill(calomet.emEtInEE());
  mCaloEmEtInHF->Fill(calomet.emEtInHF());
}

void METTester::fillGenMET(const reco::GenMET &genmet) {
  mNeutralEMEtFraction->Fill(genmet.NeutralEMEtFraction());
  mNeutralHadEtFraction->Fill(genmet.NeutralHadEtFraction());
  mChargedEMEtFraction->Fill(genmet.ChargedEMEtFraction());
  mChargedHadEtFraction->Fill(genmet.ChargedHadEtFraction());
  mMuonEtFraction->Fill(genmet.MuonEtFraction());
  mInvisibleEtFraction->Fill(genmet.InvisibleEtFraction());
}

void METTester::fillPFMET(const reco::PFMET &pfmet) {
  mPFphotonEtFraction->Fill(pfmet.photonEtFraction());
  mPFphotonEt->Fill(pfmet.photonEt());
  mPFneutralHadronEtFraction->Fill(pfmet.neutralHadronEtFraction());
  mPFneutralHadronEt->Fill(pfmet.neutralHadronEt());
  mPFelectronEtFraction->Fill(pfmet.electronEtFraction());
  mPFelectronEt->Fill(pfmet.electronEt());
  mPFchargedHadronEtFraction->Fill(pfmet.chargedHadronEtFraction());
  mPFchargedHadronEt->Fill(pfmet.chargedHadronEt());
  mPFmuonEtFraction->Fill(pfmet.muonEtFraction());
  mPFmuonEt->Fill(pfmet.muonEt());
  mPFHFHadronEtFraction->Fill(pfmet.HFHadronEtFraction());  // HF Hadronic energy fraction
  mPFHFHadronEt->Fill(pfmet.HFHadronEt());                  // HF Hadronic energy
  mPFHFEMEtFraction->Fill(pfmet.HFEMEtFraction());          // HF EM energy fraction
  mPFHFEMEt->Fill(pfmet.HFEMEt());                          // HF EM energy
}

void METTester::fillPatMET(const pat::MET &patmet, double met) {
  mMETUnc_JetResUp->Fill(met - patmet.shiftedPt(pat::MET::JetResUp));
  mMETUnc_JetResDown->Fill(met - patmet.shiftedPt(pat::MET::JetResDown));
  mMETUnc_JetEnUp->Fill(met - patmet.shiftedPt(pat::MET::JetEnUp));
  mMETUnc_JetEnDown->Fill(met - patmet.shiftedPt(pat::MET::JetEnDown));
  mMETUnc_MuonEnUp->Fill(met - patmet.shiftedPt(pat::MET::MuonEnUp));
  mMETUnc_MuonEnDown->Fill(met - patmet.shiftedPt(pat::MET::MuonEnDown));
  mMETUnc_ElectronEnUp->Fill(met - patmet.shiftedPt(pat::MET::ElectronEnUp));
  mMETUnc_ElectronEnDown->Fill(met - patmet.shiftedPt(pat::MET::ElectronEnDown));
  mMETUnc_TauEnUp->Fill(met - patmet.shiftedPt(pat::MET::TauEnUp));
  mMETUnc_TauEnDown->Fill(met - patmet.shiftedPt(pat::MET::TauEnDown));
  mMETUnc_UnclusteredEnUp->Fill(met - patmet.shiftedPt(pat::MET::UnclusteredEnUp));
  mMETUnc_UnclusteredEnDown->Fill(met - patmet.shiftedPt(pat::MET::UnclusteredEnDown));
  mMETUnc_PhotonEnUp->Fill(met - patmet.shiftedPt(pat::MET::PhotonEnUp));
  mMETUnc_PhotonEnDown->Fill(met - patmet.shiftedPt(pat::MET::PhotonEnDown));

  if (patmet.isPFMET()) {
    mPFphotonEtFraction->Fill(patmet.NeutralEMFraction());
    mPFneutralHadronEtFraction->Fill(patmet.NeutralHadEtFraction());
    mPFelectronEtFraction->Fill(patmet.ChargedEMEtFraction());
    mPFchargedHadronEtFraction->Fill(patmet.ChargedHadEtFraction());
    mPFmuonEtFraction->Fill(patmet.MuonEtFraction());
    mPFHFHadronEtFraction->Fill(patmet.Type6EtFraction());  // HF Hadronic energy fraction
    mPFHFEMEtFraction->Fill(patmet.Type7EtFraction());      // HF EM energy fraction
  }
}

//------------------------------------------------------------------------------
// fill description
//------------------------------------------------------------------------------
void METTester::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  // Default MET validation offline
  desc.addUntracked<std::string>("runDir", "JetMET/METValidation/");
  desc.addUntracked<std::string>("outFolder", "")
      ->setComment("Output folder, relative to runDir. If empty, the module label is used.");
  desc.add<edm::InputTag>("primaryVertices", edm::InputTag("PixelVertices"));
  desc.add<std::string>("inputMETLabel", "pfMet")
      ->setComment("MET to validate; for METType 'mht', optional MET companion for the MET vs MHT plots.");
  desc.add<std::string>("inputMHTLabel", "")
      ->setComment(
          "MHT to validate for METType 'mht'; for other types, optional MHT companion for the MET vs MHT plots. "
          "Configure a given MET-MHT pair in one module only.");
  desc.addUntracked<std::string>("METType", "pf")->setComment("One of: calo, pf, mht, gen, miniaod.");
  desc.add<std::string>("genMetTrueLabel", "genMetTrue");
  desc.add<std::string>("genMetCaloLabel", "genMetCalo");
  desc.add<std::string>("genMhtLabel", "genMht");
  descriptions.addWithDefaultLabel(desc);
}
