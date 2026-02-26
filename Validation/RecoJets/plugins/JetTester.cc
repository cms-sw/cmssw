// Producer for validation histograms for Calo, JPT and PF jet objects
// F. Ratnikov, Sept. 7, 2006
// Modified by Chiyoung Jeong, Feb. 2, 2010
// Modified by J. Piedra, Sept. 11, 2013
// Modified by E. Vernazza, Aug. 1, 2025

#include "JetTester.h"
#include <vector>
#include <format>

using namespace edm;
using namespace reco;
using namespace std;

JetTester::JetTester(const edm::ParameterSet& iConfig)
    : mInputCollection(iConfig.getParameter<edm::InputTag>("src")),
      //  rhoTag                         (iConfig.getParameter<edm::InputTag>
      //  ("srcRho")),
      JetType(iConfig.getUntrackedParameter<std::string>("JetType")),
      mRecoJetPtThreshold(iConfig.getParameter<double>("recoJetPtThreshold")),
      mMatchGenPtThreshold(iConfig.getParameter<double>("matchGenPtThreshold")),
      mRThreshold(iConfig.getParameter<double>("RThreshold")) {
  std::string inputCollectionLabel(mInputCollection.label());
  isHLT_ = iConfig.getUntrackedParameter<bool>("isHLT", false);

  // Flag for the definition of Jet Input Collection
  isCaloJet = (std::string("calo") == JetType);        // <reco::CaloJetCollection>
  isPFJet = (std::string("pf") == JetType);            // <reco::PFJetCollection>
  isMiniAODJet = (std::string("miniaod") == JetType);  // <pat::JetCollection>
  if (!isCaloJet && !isPFJet && !isMiniAODJet) {
    throw cms::Exception("Configuration")
        << "Unknown jet type: " << JetType << "\nPlease use 'calo', 'pf', or 'miniaod'.";
  }

  if (!isMiniAODJet) {
    mJetCorrector = iConfig.getParameter<edm::InputTag>("JetCorrections");
  }

  // consumes
  pvToken_ = consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("primVertex"));
  if (isCaloJet)
    caloJetsToken_ = consumes<reco::CaloJetCollection>(mInputCollection);
  if (isPFJet)
    pfJetsToken_ = consumes<reco::PFJetCollection>(mInputCollection);
  if (isMiniAODJet)
    patJetsToken_ = consumes<pat::JetCollection>(mInputCollection);
  mInputGenCollection = iConfig.getParameter<edm::InputTag>("srcGen");
  genJetsToken_ = consumes<reco::GenJetCollection>(edm::InputTag(mInputGenCollection));
  evtToken_ = consumes<GenEventInfoProduct>(edm::InputTag("generator"));
  if (!isMiniAODJet && !mJetCorrector.label().empty()) {
    jetCorrectorToken_ = consumes<reco::JetCorrector>(mJetCorrector);
  }

  // Events variables
  mNvtx = nullptr;

  // Jet parameters
  mJetEta = nullptr;
  mJetPhi = nullptr;
  mJetPt = nullptr;
  mJetEtaPt = nullptr;
  mJetPhiPt = nullptr;
  mJetEnergy = nullptr;
  mJetMass = nullptr;
  mJetConstituents = nullptr;
  mJetArea = nullptr;

  // Corrected jet parameters
  mCorrJetEta = nullptr;
  mCorrJetPhi = nullptr;
  mCorrJetPt = nullptr;
  mCorrJetEtaPt = nullptr;
  mCorrJetPhiPt = nullptr;

  // Gen jet parameters
  mGenEta = nullptr;
  mGenPhi = nullptr;
  mGenPt = nullptr;
  mGenEtaPt = nullptr;
  mGenPhiPt = nullptr;

  // Matched jet parameters
  mMatchedJetEta = nullptr;
  mMatchedJetPhi = nullptr;
  mMatchedJetPt = nullptr;
  mMatchedJetEtaPt = nullptr;
  mMatchedJetPhiPt = nullptr;
  mMatchedJetchHad = nullptr;
  mMatchedJetneHad = nullptr;
  mMatchedJetchEm = nullptr;
  mMatchedJetneEm = nullptr;
  mMatchedJetnCost = nullptr;

  // Matched gen jet parameters
  mMatchedGenEta = nullptr;
  mMatchedGenPhi = nullptr;
  mMatchedGenPt = nullptr;
  mMatchedGenEtaPt = nullptr;
  mMatchedGenPhiPt = nullptr;

  // First jet parameters
  mJetEtaFirst = nullptr;
  mJetPhiFirst = nullptr;
  mJetPtFirst = nullptr;
  mGenEtaFirst = nullptr;
  mGenPhiFirst = nullptr;
  mGenPtFirst = nullptr;

  // Other variables
  mMjj = nullptr;
  mNJets = nullptr;
  mNJetsPt1 = nullptr;
  mNJetsPt2 = nullptr;
  mDeltaEta = nullptr;
  mDeltaPhi = nullptr;
  mDeltaPt = nullptr;

  // ---- Calo Jet specific information ----
  /// returns the maximum energy deposited in ECAL towers
  maxEInEmTowers = nullptr;
  /// returns the maximum energy deposited in HCAL towers
  maxEInHadTowers = nullptr;
  /// returns the jet hadronic energy fraction
  energyFractionHadronic = nullptr;
  /// returns the jet electromagnetic energy fraction
  emEnergyFraction = nullptr;
  /// returns the jet hadronic energy in HB
  hadEnergyInHB = nullptr;
  /// returns the jet hadronic energy in HO
  hadEnergyInHO = nullptr;
  /// returns the jet hadronic energy in HE
  hadEnergyInHE = nullptr;
  /// returns the jet hadronic energy in HF
  hadEnergyInHF = nullptr;
  /// returns the jet electromagnetic energy in EB
  emEnergyInEB = nullptr;
  /// returns the jet electromagnetic energy in EE
  emEnergyInEE = nullptr;
  /// returns the jet electromagnetic energy extracted from HF
  emEnergyInHF = nullptr;
  /// returns area of contributing towers
  towersArea = nullptr;
  /// returns the number of constituents carrying a 90% of the total Jet
  /// energy*/
  n90 = nullptr;
  /// returns the number of constituents carrying a 60% of the total Jet
  /// energy*/
  n60 = nullptr;

  // ---- JPT Jet specific information ----
  /// chargedMultiplicity
  //  elecMultiplicity = 0;

  // ---- JPT or PF Jet specific information ----
  /// muonMultiplicity
  muonMultiplicity = nullptr;
  /// chargedMultiplicity
  chargedMultiplicity = nullptr;
  /// chargedEmEnergy
  chargedEmEnergy = nullptr;
  /// neutralEmEnergy
  neutralEmEnergy = nullptr;
  /// chargedHadronEnergy
  chargedHadronEnergy = nullptr;
  /// neutralHadronEnergy
  neutralHadronEnergy = nullptr;
  /// chargedHadronEnergyFraction (relative to uncorrected jet energy)
  chargedHadronEnergyFraction = nullptr;
  /// neutralHadronEnergyFraction (relative to uncorrected jet energy)
  neutralHadronEnergyFraction = nullptr;
  /// chargedEmEnergyFraction (relative to uncorrected jet energy)
  chargedEmEnergyFraction = nullptr;
  /// neutralEmEnergyFraction (relative to uncorrected jet energy)
  neutralEmEnergyFraction = nullptr;

  // ---- PF Jet specific information ----
  /// photonEnergy
  photonEnergy = nullptr;
  /// photonEnergyFraction (relative to corrected jet energy)
  photonEnergyFraction = nullptr;
  /// electronEnergy
  electronEnergy = nullptr;
  /// electronEnergyFraction (relative to corrected jet energy)
  electronEnergyFraction = nullptr;
  /// muonEnergy
  muonEnergy = nullptr;
  /// muonEnergyFraction (relative to corrected jet energy)
  muonEnergyFraction = nullptr;
  /// HFHadronEnergy
  HFHadronEnergy = nullptr;
  /// HFHadronEnergyFraction (relative to corrected jet energy)
  HFHadronEnergyFraction = nullptr;
  /// HFEMEnergy
  HFEMEnergy = nullptr;
  /// HFEMEnergyFraction (relative to corrected jet energy)
  HFEMEnergyFraction = nullptr;

  chargedHadronMultiplicity = nullptr;
  neutralHadronMultiplicity = nullptr;
  photonMultiplicity = nullptr;
  electronMultiplicity = nullptr;
  HFHadronMultiplicity = nullptr;
  HFEMMultiplicity = nullptr;
  chargedMuEnergy = nullptr;
  chargedMuEnergyFraction = nullptr;
  neutralMultiplicity = nullptr;

  for (size_t j = 0; j < etaInfo.size(); ++j) {
    photonMultiplicity_EtaBins[j] = nullptr;
    electronMultiplicity_EtaBins[j] = nullptr;
    neutralHadronMultiplicity_EtaBins[j] = nullptr;
    neutralMultiplicity_EtaBins[j] = nullptr;
    chargedHadronMultiplicity_EtaBins[j] = nullptr;
    chargedMultiplicity_EtaBins[j] = nullptr;
  }

  /// HOEnergy
  HOEnergy = nullptr;
  /// HOEnergyFraction (relative to corrected jet energy)
  HOEnergyFraction = nullptr;

  // contained in MiniAOD
  hadronFlavor = nullptr;
  partonFlavor = nullptr;
  genPartonPDGID = nullptr;
}

void JetTester::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const&) {
  if (isHLT_)
    ibooker.setCurrentFolder("HLT/JetMET/JetValidation/" + mInputCollection.label());
  else
    ibooker.setCurrentFolder("JetMET/JetValidation/" + mInputCollection.label());

  // Discard all reco and corrected jets below this min threshold
  minJetPt = 20.;

  int n_EtaBins = 60;
  std::vector<double> EtaRange = {-6.0, 6.0};
  int n_PhiBins = 70;
  std::vector<double> PhiRange = {-3.5, 3.5};
  int n_PtBins = 50;
  std::vector<double> PtRange = {0, 1000};
  int n_RespBins = 70;
  std::vector<double> RespRange = {0, 3.5};

  // Event variables
  mNvtx = ibooker.book1D("Nvtx", "number of vertices", 60, 0, 60);

  // Jet parameters
  mJetEta = ibooker.book1D("JetEta",
                           "Reco Jets p_{T}>" + std::to_string(int(mRecoJetPtThreshold)) + " GeV;#eta;# jets",
                           n_EtaBins,
                           EtaRange[0],
                           EtaRange[1]);
  mJetPhi = ibooker.book1D("JetPhi",
                           "Reco Jets p_{T}>" + std::to_string(int(mRecoJetPtThreshold)) + " GeV;#phi;# jets",
                           n_PhiBins,
                           PhiRange[0],
                           PhiRange[1]);
  mJetPt = ibooker.book1D("JetPt",
                          "Reco Jets p_{T}>" + std::to_string(int(mRecoJetPtThreshold)) + " GeV;p_{T} [GeV];# jets",
                          n_PtBins,
                          PtRange[0],
                          PtRange[1]);
  mJetEtaPt = ibooker.book2D("JetEtaPt",
                             "Reco Jets p_{T}>" + std::to_string(int(mRecoJetPtThreshold)) + " GeV;#eta;p_{T}",
                             n_EtaBins,
                             EtaRange[0],
                             EtaRange[1],
                             n_PtBins,
                             PtRange[0],
                             PtRange[1]);
  mJetPhiPt = ibooker.book2D("JetPhiPt",
                             "Reco Jets p_{T}>" + std::to_string(int(mRecoJetPtThreshold)) + " GeV;#phi;p_{T}",
                             n_PhiBins,
                             PhiRange[0],
                             PhiRange[1],
                             n_PtBins,
                             PtRange[0],
                             PtRange[1]);
  mJetEnergy = ibooker.book1D("JetEnergy", "Reco Jets;Energy [GeV];# jets", n_PtBins, PtRange[0], PtRange[1]);
  mJetMass = ibooker.book1D("JetMass", "Reco Jets;Mass [GeV];# jets", 100, 0, 200);
  mJetConstituents = ibooker.book1D("JetConstituents", "Reco Jets;# constituents;# jets", 100, 0, 100);
  mJetArea = ibooker.book1D("JetArea", "Reco Jets;Area;# jets", 100, 0, 4);

  // Gen jet parameters
  mGenEta = ibooker.book1D("GenEta",
                           "Gen Jets p_{T}>" + std::to_string(int(mMatchGenPtThreshold)) + " GeV;#eta;# jets",
                           n_EtaBins,
                           EtaRange[0],
                           EtaRange[1]);
  mGenPhi = ibooker.book1D("GenPhi",
                           "Gen Jets p_{T}>" + std::to_string(int(mMatchGenPtThreshold)) + " GeV;#phi;# jets",
                           n_PhiBins,
                           PhiRange[0],
                           PhiRange[1]);
  mGenPt = ibooker.book1D("GenPt",
                          "Gen Jets p_{T}>" + std::to_string(int(mMatchGenPtThreshold)) + " GeV;p_{T};# jets",
                          n_PtBins,
                          PtRange[0],
                          PtRange[1]);
  mGenEtaPt = ibooker.book2D("GenEtaPt",
                             "Gen Jets p_{T}>" + std::to_string(int(mMatchGenPtThreshold)) + " GeV;#eta;p_{T}",
                             n_EtaBins,
                             EtaRange[0],
                             EtaRange[1],
                             n_PtBins,
                             PtRange[0],
                             PtRange[1]);
  mGenPhiPt = ibooker.book2D("GenPhiPt",
                             "Gen Jets p_{T}>" + std::to_string(int(mMatchGenPtThreshold)) + " GeV;#phi;p_{T}",
                             n_PhiBins,
                             PhiRange[0],
                             PhiRange[1],
                             n_PtBins,
                             PtRange[0],
                             PtRange[1]);

  // Matched jet parameters
  mMatchedJetEta = ibooker.book1D("MatchedJetEta", "Matched Jets;#eta;# jets", n_EtaBins, EtaRange[0], EtaRange[1]);
  mMatchedJetPhi = ibooker.book1D("MatchedJetPhi", "Matched Jets;#phi;# jets", n_PhiBins, PhiRange[0], PhiRange[1]);
  mMatchedJetPt = ibooker.book1D("MatchedJetPt", "Matched Jets;p_{T};# jets", n_PtBins, PtRange[0], PtRange[1]);
  mMatchedJetchHad = ibooker.book1D("MatchedJetchHad", "Matched Jets;charged HAD energy fraction;# jets", 50, 0, 1);
  mMatchedJetneHad = ibooker.book1D("MatchedJetneHad", "Matched Jets;neutral HAD energy fraction;# jets", 50, 0, 1);
  mMatchedJetchEm = ibooker.book1D("MatchedJetchEm", "Matched Jets;charged EM energy fraction;# jets", 50, 0, 1);
  mMatchedJetneEm = ibooker.book1D("MatchedJetneEm", "Matched Jets;neutral EM energy fraction;# jets", 50, 0, 1);
  mMatchedJetnCost = ibooker.book1D("MatchedJetnCost", "Matched Jets;# constituents;# jets", 100, 0, 100);
  mMatchedJetEtaPt = ibooker.book2D("MatchedJetEtaPt",
                                    "Matched Jets;#eta;p_{T}",
                                    n_EtaBins,
                                    EtaRange[0],
                                    EtaRange[1],
                                    n_PtBins,
                                    PtRange[0],
                                    PtRange[1]);
  mMatchedJetPhiPt = ibooker.book2D("MatchedJetPhiPt",
                                    "Matched Jets;#[hi];p_{T}",
                                    n_PhiBins,
                                    PhiRange[0],
                                    PhiRange[1],
                                    n_PtBins,
                                    PtRange[0],
                                    PtRange[1]);

  // DeltaR distance jet
  h2d_DeltaR_vs_Eta = ibooker.book2D(
      "h2d_DeltaR_Eta", "#Delta R Jets;#eta^{reco};#Delta R", n_EtaBins, EtaRange[0], EtaRange[1], 50, 0.0, 0.5);
  h2d_DeltaR_vs_Phi = ibooker.book2D(
      "h2d_DeltaR_Phi", "#Delta R Jets;#phi^{reco};#Delta R", n_PhiBins, PhiRange[0], PhiRange[1], 50, 0.0, 0.5);
  h2d_DeltaR_vs_Pt = ibooker.book2D(
      "h2d_DeltaR_Pt", "#Delta R Jets;p_{T}^{reco};#Delta R", n_PtBins, PtRange[0], PtRange[1], 50, 0.0, 0.5);

  // Duplicated jet
  mDuplicatesJetEta = ibooker.book1D(
      "DuplicatesJetEta", "Gen jets matched to multiple reco jet;#eta;# gen jets", n_EtaBins, EtaRange[0], EtaRange[1]);
  mDuplicatesJetPhi = ibooker.book1D(
      "DuplicatesJetPhi", "Gen jets matched to multiple reco jet;#phi;# gen jets", n_PhiBins, PhiRange[0], PhiRange[1]);
  mDuplicatesJetPt = ibooker.book1D("DuplicatesJetPt",
                                    "Gen jets matched to multiple reco jets;p_{T} [GeV];# gen jets",
                                    n_PtBins,
                                    PtRange[0],
                                    PtRange[1]);
  mDuplicatesJetEtaPt = ibooker.book2D("DuplicatesJetEtaPt",
                                       "Gen jets matched to multiple reco jets;#eta;p_{T} [GeV]",
                                       n_EtaBins,
                                       EtaRange[0],
                                       EtaRange[1],
                                       n_PtBins,
                                       PtRange[0],
                                       PtRange[1]);
  mDuplicatesJetPhiPt = ibooker.book2D("DuplicatesJetPhiPt",
                                       "Gen jets matched to multiple reco jets;#phi;p_{T} [GeV]",
                                       n_PhiBins,
                                       PhiRange[0],
                                       PhiRange[1],
                                       n_PtBins,
                                       PtRange[0],
                                       PtRange[1]);

  // Matched gen jet parameters
  mMatchedGenEta = ibooker.book1D("MatchedGenEta", "Matched Gen Jets;#eta;# jets", n_EtaBins, EtaRange[0], EtaRange[1]);
  mMatchedGenPhi = ibooker.book1D("MatchedGenPhi", "Matched Gen Jets;#phi;# jets", n_PhiBins, PhiRange[0], PhiRange[1]);
  mMatchedGenPt = ibooker.book1D("MatchedGenPt", "Matched Gen Jets;p_{T};# jets", n_PtBins, PtRange[0], PtRange[1]);
  mMatchedGenEtaPt = ibooker.book2D("MatchedGenEtaPt",
                                    "Matched Gen Jets;#eta;p_{T}",
                                    n_EtaBins,
                                    EtaRange[0],
                                    EtaRange[1],
                                    n_PtBins,
                                    PtRange[0],
                                    PtRange[1]);
  mMatchedGenPhiPt = ibooker.book2D("MatchedGenPhiPt",
                                    "Matched Gen Jets;#[hi];p_{T}",
                                    n_PhiBins,
                                    PhiRange[0],
                                    PhiRange[1],
                                    n_PtBins,
                                    PtRange[0],
                                    PtRange[1]);

  // DeltaR distance gen jet
  h2d_DeltaR_vs_GenEta = ibooker.book2D(
      "h2d_DeltaR_GenEta", "#Delta R Gen Jets;#eta^{gen};#Delta R", n_EtaBins, EtaRange[0], EtaRange[1], 50, 0.0, 0.5);
  h2d_DeltaR_vs_GenPhi = ibooker.book2D(
      "h2d_DeltaR_GenPhi", "#Delta R Gen Jets;#phi^{gen};#Delta R", n_PhiBins, PhiRange[0], PhiRange[1], 50, 0.0, 0.5);
  h2d_DeltaR_vs_GenPt = ibooker.book2D(
      "h2d_DeltaR_GenPt", "#Delta R Gen Jets;p_{T}^{gen};#Delta R", n_PtBins, PtRange[0], PtRange[1], 50, 0.0, 0.5);

  // Duplicated gen jet
  mDuplicatesGenEta = ibooker.book1D(
      "DuplicatesGenEta", "Reco jets matched to multiple gen jets;#eta;# jets", n_EtaBins, EtaRange[0], EtaRange[1]);
  mDuplicatesGenPhi = ibooker.book1D(
      "DuplicatesGenPhi", "Reco jets matched to multiple gen jets;#phi;# jets", n_PhiBins, PhiRange[0], PhiRange[1]);
  mDuplicatesGenPt = ibooker.book1D(
      "DuplicatesGenPt", "Reco jets matched to multiple gen jets;p_{T} [GeV];# jets", n_PtBins, PtRange[0], PtRange[1]);
  mDuplicatesGenEtaPt = ibooker.book2D("DuplicatesGenEtaPt",
                                       "Reco jets matched to multiple gen jets;#eta;p_{T} [GeV]",
                                       n_EtaBins,
                                       EtaRange[0],
                                       EtaRange[1],
                                       n_PtBins,
                                       PtRange[0],
                                       PtRange[1]);
  mDuplicatesGenPhiPt = ibooker.book2D("DuplicatesGenPhiPt",
                                       "Reco jets matched to multiple gen jets;#phi;p_{T} [GeV]",
                                       n_PhiBins,
                                       PhiRange[0],
                                       PhiRange[1],
                                       n_PtBins,
                                       PtRange[0],
                                       PtRange[1]);

  // Response
  h2d_JetPtRecoOverGen_vs_GenEta = ibooker.book2D("h2d_PtRecoOverGen_GenEta",
                                                  "Response Reco Jets;#eta^{gen};p_{T}^{reco}/p_{T}^{gen}",
                                                  n_EtaBins,
                                                  EtaRange[0],
                                                  EtaRange[1],
                                                  n_RespBins,
                                                  RespRange[0],
                                                  RespRange[1]);
  h2d_JetPtRecoOverGen_vs_GenPhi = ibooker.book2D("h2d_PtRecoOverGen_GenPhi",
                                                  "Response Reco Jets;#phi^{gen};p_{T}^{reco}/p_{T}^{gen}",
                                                  n_PhiBins,
                                                  PhiRange[0],
                                                  PhiRange[1],
                                                  n_RespBins,
                                                  RespRange[0],
                                                  RespRange[1]);
  h2d_JetPtRecoOverGen_vs_GenPt = ibooker.book2D("h2d_PtRecoOverGen_GenPt",
                                                 "Response Reco Jets;p_{T}^{gen};p_{T}^{reco}/p_{T}^{gen}",
                                                 n_PtBins,
                                                 PtRange[0],
                                                 PtRange[1],
                                                 n_RespBins,
                                                 RespRange[0],
                                                 RespRange[1]);

  for (size_t j = 0; j < etaInfo.size(); ++j) {
    const auto& [etaRegion, etaLabel, etaMin, etaMax] = etaInfo[j];

    mNJets_EtaBins[j] = ibooker.book1D(
        std::format("NJets_{}", etaRegion),
        std::format("Number of jets p_{{T}}>{} GeV - {};# jets;# events", int(mRecoJetPtThreshold), etaLabel),
        15,
        0,
        15);
    mGenPt_EtaBins[j] = ibooker.book1D(
        std::format("GenPt_{}", etaRegion),
        std::format("Gen Jets p_{{T}}>{} GeV - {};p_{{T}} [GeV];# jets", int(mMatchGenPtThreshold), etaLabel),
        n_PtBins,
        PtRange[0],
        PtRange[1]);

    mJetPt_EtaBins[j] = ibooker.book1D(
        std::format("JetPt_{}", etaRegion),
        std::format("Reco Jets p_{{T}}>{} GeV - {};p_{{T}} [GeV];# jets", int(mRecoJetPtThreshold), etaLabel),
        n_PtBins,
        PtRange[0],
        PtRange[1]);

    mMatchedJetPt_EtaBins[j] = ibooker.book1D(std::format("MatchedJetPt_{}", etaRegion),
                                              std::format("Matched Jets - {};p_{{T}} [GeV];# jets", etaLabel),
                                              n_PtBins,
                                              PtRange[0],
                                              PtRange[1]);
    mMatchedGenPt_EtaBins[j] = ibooker.book1D(std::format("MatchedGenPt_{}", etaRegion),
                                              std::format("Matched Gen Jets - {};p_{{T}} [GeV];# jets", etaLabel),
                                              n_PtBins,
                                              PtRange[0],
                                              PtRange[1]);

    mDuplicatesGenPt_EtaBins[j] =
        ibooker.book1D(std::format("DuplicatesGenPt_{}", etaRegion),
                       std::format("Reco jets matched to multiple gen jets [{}];p_{{T}} [GeV];# Duplicates", etaLabel),
                       n_PtBins,
                       PtRange[0],
                       PtRange[1]);
    mDuplicatesJetPt_EtaBins[j] =
        ibooker.book1D(std::format("DuplicatesJetPt_{}", etaRegion),
                       std::format("Gen jets matched to multiple reco jets [{}];p_{{T}} [GeV];# Duplicates", etaLabel),
                       n_PtBins,
                       PtRange[0],
                       PtRange[1]);

    h_JetPtRecoOverGen[j] =
        ibooker.book1D(std::format("h_PtRecoOverGen_{}", etaRegion),
                       std::format("Response Reco Jets - {};p_{{T}}^{{reco}}/p_{{T}}^{{gen}};# jets", etaLabel),
                       n_RespBins,
                       RespRange[0],
                       RespRange[1]);

    h2d_JetPtRecoOverGen_vs_GenPhi_EtaBins[j] =
        ibooker.book2D(std::format("h2d_PtRecoOverGen_GenPhi_{}", etaRegion),
                       std::format("Response Reco Jets - {};#phi^{{gen}};p_{{T}}^{{reco}}/p_{{T}}^{{gen}}", etaLabel),
                       n_PhiBins,
                       PhiRange[0],
                       PhiRange[1],
                       n_RespBins,
                       RespRange[0],
                       RespRange[1]);
    h2d_JetPtRecoOverGen_vs_GenPt_EtaBins[j] = ibooker.book2D(
        std::format("h2d_PtRecoOverGen_GenPt_{}", etaRegion),
        std::format("Response Reco Jets - {};p_{{T}}^{{gen}};p_{{T}}^{{reco}}/p_{{T}}^{{gen}}", etaLabel),
        n_PtBins,
        PtRange[0],
        PtRange[1],
        n_RespBins,
        RespRange[0],
        RespRange[1]);

    h2d_chHadMult_vs_pt[j] = ibooker.book2D(
        std::format("h2d_chHadMult_pt_{}", etaRegion),
        std::format("Profiled charged HAD multiplicity - {};p_{{T}}^{{reco}};charged HAD multiplicity", etaLabel),
        n_PtBins,
        PtRange[0],
        PtRange[1],
        50,
        0,
        50);
    h2d_neHadMult_vs_pt[j] = ibooker.book2D(
        std::format("h2d_neHadMult_pt_{}", etaRegion),
        std::format("Profiled neutral HAD multiplicity - {};p_{{T}}^{{reco}};neutral HAD multiplicity", etaLabel),
        n_PtBins,
        PtRange[0],
        PtRange[1],
        50,
        0,
        50);
    h2d_chMult_vs_pt[j] = ibooker.book2D(
        std::format("h2d_chMult_pt_{}", etaRegion),
        std::format("Profiled charged multiplicity - {};p_{{T}}^{{reco}};charged multiplicity", etaLabel),
        n_PtBins,
        PtRange[0],
        PtRange[1],
        50,
        0,
        50);
    h2d_neMult_vs_pt[j] = ibooker.book2D(
        std::format("h2d_neMult_pt_{}", etaRegion),
        std::format("Profiled neutral multiplicity - {};p_{{T}}^{{reco}};neutral EM multiplicity", etaLabel),
        n_PtBins,
        PtRange[0],
        PtRange[1],
        50,
        0,
        50);
    h2d_phoMult_vs_pt[j] =
        ibooker.book2D(std::format("h2d_phoMult_pt_{}", etaRegion),
                       std::format("Profiled photon multiplicity - {};p_{{T}}^{{reco}};photon multiplicity", etaLabel),
                       n_PtBins,
                       PtRange[0],
                       PtRange[1],
                       15,
                       0,
                       15);

    h2d_chHad_vs_pt[j] = ibooker.book2D(
        std::format("h2d_chHad_pt_{}", etaRegion),
        std::format("Profiled charged HAD energy fraction - {};p_{{T}}^{{reco}};charged HAD energy fraction", etaLabel),
        n_PtBins,
        PtRange[0],
        PtRange[1],
        40,
        0,
        1);
    h2d_neHad_vs_pt[j] = ibooker.book2D(
        std::format("h2d_neHad_pt_{}", etaRegion),
        std::format("Profiled neutral HAD energy fraction - {};p_{{T}}^{{reco}};neutral HAD energy fraction", etaLabel),
        n_PtBins,
        PtRange[0],
        PtRange[1],
        40,
        0,
        1);
    h2d_chEm_vs_pt[j] = ibooker.book2D(
        std::format("h2d_chEm_pt_{}", etaRegion),
        std::format("Profiled charged EM energy fraction - {};p_{{T}}^{{reco}};charged EM energy fraction", etaLabel),
        n_PtBins,
        PtRange[0],
        PtRange[1],
        40,
        0,
        1);
    h2d_neEm_vs_pt[j] = ibooker.book2D(
        std::format("h2d_neEm_pt_{}", etaRegion),
        std::format("Profiled neutral EM energy fraction - {};p_{{T}}^{{reco}};neutral EM energy fraction", etaLabel),
        n_PtBins,
        PtRange[0],
        PtRange[1],
        40,
        0,
        1);

    h2d_JetPtRecoOverGen_vs_chHad[j] = ibooker.book2D(
        std::format("h2d_PtRecoOverGen_chHad_{}", etaRegion),
        std::format("Response Reco Jets - {};charged HAD energy Fraction;p_{{T}}^{{reco}}/p_{{T}}^{{gen}}", etaLabel),
        30,
        0,
        1,
        n_RespBins,
        RespRange[0],
        RespRange[1]);
    h2d_JetPtRecoOverGen_vs_neHad[j] = ibooker.book2D(
        std::format("h2d_PtRecoOverGen_neHad_{}", etaRegion),
        std::format("Response Reco Jets - {};neutral HAD energy Fraction;p_{{T}}^{{reco}}/p_{{T}}^{{gen}}", etaLabel),
        30,
        0,
        1,
        n_RespBins,
        RespRange[0],
        RespRange[1]);
    h2d_JetPtRecoOverGen_vs_chEm[j] = ibooker.book2D(
        std::format("h2d_PtRecoOverGen_chEm_{}", etaRegion),
        std::format("Response Reco Jets - {};charged EM energy Fraction;p_{{T}}^{{reco}}/p_{{T}}^{{gen}}", etaLabel),
        30,
        0,
        1,
        n_RespBins,
        RespRange[0],
        RespRange[1]);
    h2d_JetPtRecoOverGen_vs_neEm[j] = ibooker.book2D(
        std::format("h2d_PtRecoOverGen_neEm_{}", etaRegion),
        std::format("Response Reco Jets - {};neutral EM energy Fraction;p_{{T}}^{{reco}}/p_{{T}}^{{gen}}", etaLabel),
        30,
        0,
        1,
        n_RespBins,
        RespRange[0],
        RespRange[1]);
    h2d_JetPtRecoOverGen_vs_nCost[j] =
        ibooker.book2D(std::format("h2d_PtRecoOverGen_nCost_{}", etaRegion),
                       std::format("Response Reco Jets - {};# constituents;p_{{T}}^{{reco}}/p_{{T}}^{{gen}}", etaLabel),
                       100,
                       0,
                       100,
                       n_RespBins,
                       RespRange[0],
                       RespRange[1]);

    for (int i = 0; i < ptSize; ++i) {
      int ptMin = int(ptBins_[i]);
      int ptMax = int(ptBins_[i + 1]);
      auto h_name_RoG = std::format("h_PtRecoOverGen_{}_Pt{}_{}", etaRegion, ptMin, ptMax);
      auto h_title_RoG =
          std::format("Response Reco Jets - {} - {}<p_{{T}}^{{gen}}<{};p_{{T}}^{{reco}}/p_{{T}}^{{gen}};# jets",
                      etaLabel,
                      ptMin,
                      ptMax);
      hVector_JetPtRecoOverGen_ptBins[j][i] =
          ibooker.book1D(h_name_RoG, h_title_RoG, n_RespBins, RespRange[0], RespRange[1]);
    }
  }

  for (int i = 0; i < ptSize; ++i) {
    int ptMin = int(ptBins_[i]);
    int ptMax = int(ptBins_[i + 1]);
    h2d_JetPtRecoOverGen_vs_GenEta_ptBins[i] = ibooker.book2D(
        std::format("h2d_PtRecoOverGen_GenEta_Pt{}_{}", ptMin, ptMax),
        std::format(
            "Response Reco Jets - {}<p_{{T}}^{{gen}}<{};#eta^{{gen}};p_{{T}}^{{reco}}/p_{{T}}^{{gen}}", ptMin, ptMax),
        n_EtaBins,
        EtaRange[0],
        EtaRange[1],
        n_RespBins,
        RespRange[0],
        RespRange[1]);
  }

  // Jet flavors contained in MiniAOD
  if (isMiniAODJet) {
    hadronFlavor = ibooker.book1D("HadronFlavor", ";Hadron Flavor;# jets", 44, -22, 22);
    partonFlavor = ibooker.book1D("PartonFlavor", ";Parton Flavor;# jets", 44, -22, 22);
    genPartonPDGID = ibooker.book1D("genPartonPDGID", ";genParton PDG ID;# jets", 44, -22, 22);
  }

  // Corrected jet parameters
  if (isMiniAODJet || !mJetCorrector.label().empty()) {
    mCorrJetEta = ibooker.book1D("CorrJetEta",
                                 "Corr Jets p_{T}>" + std::to_string(int(mRecoJetPtThreshold)) + " GeV;#eta;# jets",
                                 n_EtaBins,
                                 EtaRange[0],
                                 EtaRange[1]);
    mCorrJetPhi = ibooker.book1D("CorrJetPhi",
                                 "Corr Jets p_{T}>" + std::to_string(int(mRecoJetPtThreshold)) + " GeV;#phi;# jets",
                                 n_PhiBins,
                                 PhiRange[0],
                                 PhiRange[1]);
    mCorrJetPt = ibooker.book1D("CorrJetPt",
                                "Corr Jets p_{T}>" + std::to_string(int(mRecoJetPtThreshold)) + " GeV;p_{T};# jets",
                                n_PtBins,
                                PtRange[0],
                                PtRange[1]);
    mCorrJetEtaPt = ibooker.book2D("CorrJetEtaPt",
                                   "Corr Jets p_{T}>" + std::to_string(int(mRecoJetPtThreshold)) + " GeV;#eta;p_{T}",
                                   n_EtaBins,
                                   EtaRange[0],
                                   EtaRange[1],
                                   n_PtBins,
                                   PtRange[0],
                                   PtRange[1]);
    mCorrJetPhiPt = ibooker.book2D("CorrJetPhiPt",
                                   "Corr Jets p_{T}>" + std::to_string(int(mRecoJetPtThreshold)) + " GeV;#phi;p_{T}",
                                   n_PhiBins,
                                   PhiRange[0],
                                   PhiRange[1],
                                   n_PtBins,
                                   PtRange[0],
                                   PtRange[1]);

    mDeltaEta = ibooker.book1D("DeltaEta", ";#eta^{gen}-#eta^{corr};# matched jets", 100, -0.5, 0.5);
    mDeltaPhi = ibooker.book1D("DeltaPhi", ";#phi^{gen}-#phi^{corr};# matched jets", 100, -0.5, 0.5);
    mDeltaPt = ibooker.book1D("DeltaPt", ";(p_{T}^{gen}-p_{T}^{corr})/p_{T}^{gen};# matched jets", 100, -1.0, 1.0);

    h2d_JetPtCorrOverGen_vs_GenEta = ibooker.book2D("h2d_PtCorrOverGen_GenEta",
                                                    "Response Corr Jets;#eta^{{gen}};p_{{T}}^{{corr}}/p_{{T}}^{{gen}}",
                                                    n_EtaBins,
                                                    EtaRange[0],
                                                    EtaRange[1],
                                                    n_RespBins,
                                                    RespRange[0],
                                                    RespRange[1]);
    h2d_JetPtCorrOverGen_vs_GenPhi = ibooker.book2D("h2d_PtCorrOverGen_GenPhi",
                                                    "Response Corr Jets;#phi^{{gen}};p_{{T}}^{{corr}}/p_{{T}}^{{gen}}",
                                                    n_PhiBins,
                                                    PhiRange[0],
                                                    PhiRange[1],
                                                    n_RespBins,
                                                    RespRange[0],
                                                    RespRange[1]);
    h2d_JetPtCorrOverGen_vs_GenPt =
        ibooker.book2D("h2d_PtCorrOverGen_GenPt",
                       "Response Corr Jets;p_{{T}}^{{gen}};p_{{T}}^{{corr}}/p_{{T}}^{{gen}}",
                       n_PtBins,
                       PtRange[0],
                       PtRange[1],
                       n_RespBins,
                       RespRange[0],
                       RespRange[1]);

    h2d_JetPtCorrOverReco_vs_Eta = ibooker.book2D("h2d_PtCorrOverReco_Eta",
                                                  "Response Corr Jets;#eta^{{reco}};p_{{T}}^{{corr}}/p_{{T}}^{{reco}}",
                                                  n_EtaBins,
                                                  EtaRange[0],
                                                  EtaRange[1],
                                                  n_RespBins,
                                                  RespRange[0],
                                                  RespRange[1]);
    h2d_JetPtCorrOverReco_vs_Phi = ibooker.book2D("h2d_PtCorrOverReco_Phi",
                                                  "Response Corr Jets;#phi^{{reco}};p_{{T}}^{{corr}}/p_{{T}}^{{reco}}",
                                                  n_PhiBins,
                                                  PhiRange[0],
                                                  PhiRange[1],
                                                  n_RespBins,
                                                  RespRange[0],
                                                  RespRange[1]);
    h2d_JetPtCorrOverReco_vs_Pt =
        ibooker.book2D("h2d_PtCorrOverReco_Pt",
                       "Response Corr Jets;p_{{T}}^{{reco}};p_{{T}}^{{corr}}/p_{{T}}^{{reco}}",
                       n_PtBins,
                       PtRange[0],
                       PtRange[1],
                       n_RespBins,
                       RespRange[0],
                       RespRange[1]);

    for (size_t j = 0; j < etaInfo.size(); ++j) {
      const auto& [etaRegion, etaLabel, etaMin, etaMax] = etaInfo[j];

      mCorrJetPt_EtaBins[j] = ibooker.book1D(
          std::format("CorrJetPt_{}", etaRegion),
          std::format("Corr Jets p_{{T}}>{} GeV - {};p_{{T}} [GeV];# jets", int(mRecoJetPtThreshold), etaLabel),
          n_PtBins,
          PtRange[0],
          PtRange[1]);
      mMatchedCorrPt_EtaBins[j] = ibooker.book1D(std::format("MatchedCorrPt_{}", etaRegion),
                                                 std::format("Matched Corr Jets - {};p_{{T}} [GeV];# jets", etaLabel),
                                                 n_PtBins,
                                                 PtRange[0],
                                                 PtRange[1]);
      h_JetPtCorrOverGen[j] =
          ibooker.book1D(std::format("h_JetPtCorrOverGen_{}", etaRegion),
                         std::format("Response Corr Jets - {};p_{{T}}^{{corr}}/p_{{T}}^{{gen}};# jets", etaLabel),
                         n_RespBins,
                         RespRange[0],
                         RespRange[1]);
      h_JetPtCorrOverReco[j] =
          ibooker.book1D(std::format("h_JetPtCorrOverReco_{}", etaRegion),
                         std::format("Response Corr Jets - {};p_{{T}}^{{corr}}/p_{{T}}^{{reco}};# jets", etaLabel),
                         n_RespBins,
                         RespRange[0],
                         RespRange[1]);
      h2d_JetPtCorrOverGen_vs_GenPhi_EtaBins[j] =
          ibooker.book2D(std::format("h2d_PtCorrOverGen_GenPhi_{}", etaRegion),
                         std::format("Response Corr Jets - {};#phi^{{gen}};p_{{T}}^{{corr}}/p_{{T}}^{{gen}}", etaLabel),
                         n_PhiBins,
                         PhiRange[0],
                         PhiRange[1],
                         n_RespBins,
                         RespRange[0],
                         RespRange[1]);
      h2d_JetPtCorrOverGen_vs_GenPt_EtaBins[j] = ibooker.book2D(
          std::format("h2d_PtCorrOverGen_GenPt_{}", etaRegion),
          std::format("Response Corr Jets - {};p_{{T}}^{{gen}};p_{{T}}^{{corr}}/p_{{T}}^{{gen}}", etaLabel),
          n_PtBins,
          PtRange[0],
          PtRange[1],
          n_RespBins,
          RespRange[0],
          RespRange[1]);
      h2d_JetPtCorrOverReco_vs_Phi_EtaBins[j] = ibooker.book2D(
          std::format("h2d_PtCorrOverReco_Phi_{}", etaRegion),
          std::format("Response Corr Jets - {};#phi^{{reco}};p_{{T}}^{{corr}}/p_{{T}}^{{reco}}", etaLabel),
          n_PhiBins,
          PhiRange[0],
          PhiRange[1],
          n_RespBins,
          RespRange[0],
          RespRange[1]);
      h2d_JetPtCorrOverReco_vs_Pt_EtaBins[j] = ibooker.book2D(
          std::format("h2d_PtCorrOverReco_Pt_{}", etaRegion),
          std::format("Response Corr Jets - {};p_{{T}}^{{reco}};p_{{T}}^{{corr}}/p_{{T}}^{{reco}}", etaLabel),
          n_PtBins,
          PtRange[0],
          PtRange[1],
          n_RespBins,
          RespRange[0],
          RespRange[1]);

      for (int i = 0; i < ptSize; ++i) {
        double ptMin = ptBins_[i];
        double ptMax = ptBins_[i + 1];
        auto h_name_CoG = std::format("h_PtCorrOverGen_{}_Pt{}_{}", etaRegion, ptMin, ptMax);
        auto h_title_CoG =
            std::format("Response Corr Jets - {} - {}<p_{{T}}^{{gen}}<{};p_{{T}}^{{corr}}/p_{{T}}^{{gen}};# jets",
                        etaLabel,
                        ptMin,
                        ptMax);
        hVector_JetPtCorrOverGen_ptBins[j][i] =
            ibooker.book1D(h_name_CoG, h_title_CoG, n_RespBins, RespRange[0], RespRange[1]);

        auto h_name_CoR = std::format("h_PtCorrOverReco_{}_Pt{}_{}", etaRegion, ptMin, ptMax);
        auto h_title_CoR =
            std::format("Response Corr Jets - {} - {}<p_{{T}}^{{reco}}<{};p_{{T}}^{{corr}}/p_{{T}}^{{reco}};# jets",
                        etaLabel,
                        ptMin,
                        ptMax);
        hVector_JetPtCorrOverReco_ptBins[j][i] =
            ibooker.book1D(h_name_CoR, h_title_CoR, n_RespBins, RespRange[0], RespRange[1]);
      }
    }

    for (int i = 0; i < ptSize; ++i) {
      int ptMin = int(ptBins_[i]);
      int ptMax = int(ptBins_[i + 1]);
      h2d_JetPtCorrOverGen_vs_GenEta_ptBins[i] = ibooker.book2D(
          std::format("h2d_PtCorrOverGen_GenEta_Pt{}_{}", ptMin, ptMax),
          std::format(
              "Response Corr Jets - {}<p_{{T}}^{{gen}}<{};#eta^{{gen}};p_{{T}}^{{corr}}/p_{{T}}^{{gen}}", ptMin, ptMax),
          n_EtaBins,
          EtaRange[0],
          EtaRange[1],
          n_RespBins,
          RespRange[0],
          RespRange[1]);
      h2d_JetPtCorrOverReco_vs_Eta_ptBins[i] = ibooker.book2D(
          std::format("h2d_PtCorrOverReco_Eta_Pt{}_{}", ptMin, ptMax),
          std::format("Response Corr Jets - {}<p_{{T}}^{{reco}}<{};#eta^{{reco}};p_{{T}}^{{corr}}/p_{{T}}^{{reco}}",
                      ptMin,
                      ptMax),
          n_EtaBins,
          EtaRange[0],
          EtaRange[1],
          n_RespBins,
          RespRange[0],
          RespRange[1]);
    }
  }

  // Generation
  mJetEtaFirst = ibooker.book1D("FirstJetEta", "First Jets;#eta;# first jets", n_EtaBins, EtaRange[0], EtaRange[1]);
  mJetPhiFirst = ibooker.book1D("FirstJetPhi", "First Jets;#phi;# first jets", n_PhiBins, PhiRange[0], PhiRange[1]);
  mJetPtFirst = ibooker.book1D("FirstJetPt", "First Jets;p_{T};# first jets", n_PtBins, PtRange[0], PtRange[1]);
  mGenEtaFirst =
      ibooker.book1D("FirstGenJetEta", "First Gen Jets;#eta;# first jets", n_EtaBins, EtaRange[0], EtaRange[1]);
  mGenPhiFirst =
      ibooker.book1D("FirstGenJetPhi", "First Gen Jets;#phi;# first jets", n_PhiBins, PhiRange[0], PhiRange[1]);
  mGenPtFirst = ibooker.book1D("FirstGenJetPt", "First Gen Jets;p_{T};# first jets", n_PtBins, PtRange[0], PtRange[1]);

  // Some jet algebra
  mMjj = ibooker.book1D("Mjj", "Mjj", 100, 0, 2000);
  mNJets = ibooker.book1D(
      "NJets", std::format("Number of jets p_{{T}}>{} GeV;# jets;# events", mRecoJetPtThreshold), 15, 0, 15);
  mNJetsPt1 = ibooker.bookProfile(
      "NJetsPt1", "Number of jets above Pt threshold;p_{T} [GeV];# jets", 100, 0, 200, 100, 0, 50, "s");
  mNJetsPt2 = ibooker.bookProfile(
      "NJetsPt2", "Number of jets above Pt threshold;p_{T} [GeV];# jets", 100, 0, 4000, 100, 0, 50, "s");

  //------------------------------------------------------------------------
  if (isCaloJet) {
    maxEInEmTowers = ibooker.book1D("maxEInEmTowers", "maxEInEmTowers", 50, 0, 500);
    maxEInHadTowers = ibooker.book1D("maxEInHadTowers", "maxEInHadTowers", 50, 0, 500);
    energyFractionHadronic = ibooker.book1D("energyFractionHadronic", "energyFractionHadronic", 50, 0, 1);
    emEnergyFraction = ibooker.book1D("emEnergyFraction", "emEnergyFraction", 50, 0, 1);
    hadEnergyInHB = ibooker.book1D("hadEnergyInHB", "hadEnergyInHB", 50, 0, 500);
    hadEnergyInHO = ibooker.book1D("hadEnergyInHO", "hadEnergyInHO", 50, 0, 500);
    hadEnergyInHE = ibooker.book1D("hadEnergyInHE", "hadEnergyInHE", 50, 0, 500);
    hadEnergyInHF = ibooker.book1D("hadEnergyInHF", "hadEnergyInHF", 50, 0, 500);
    emEnergyInEB = ibooker.book1D("emEnergyInEB", "emEnergyInEB", 50, 0, 500);
    emEnergyInEE = ibooker.book1D("emEnergyInEE", "emEnergyInEE", 50, 0, 500);
    emEnergyInHF = ibooker.book1D("emEnergyInHF", "emEnergyInHF", 50, 0, 500);
    towersArea = ibooker.book1D("towersArea", "towersArea", 50, 0, 1);
    n90 = ibooker.book1D("n90", "n90", 30, 0, 30);
    n60 = ibooker.book1D("n60", "n60", 30, 0, 30);
  }

  if (isPFJet || isMiniAODJet) {
    muonMultiplicity = ibooker.book1D("muonMultiplicity", "muonMultiplicity", 10, 0, 10);
    chargedMultiplicity = ibooker.book1D("chargedMultiplicity", "chargedMultiplicity", 100, 0, 100);
    chargedEmEnergy = ibooker.book1D("chargedEmEnergy", "chargedEmEnergy", 100, 0, 500);
    neutralEmEnergy = ibooker.book1D("neutralEmEnergy", "neutralEmEnergy", 100, 0, 500);
    chargedHadronEnergy = ibooker.book1D("chargedHadronEnergy", "chargedHadronEnergy", 100, 0, 500);
    neutralHadronEnergy = ibooker.book1D("neutralHadronEnergy", "neutralHadronEnergy", 100, 0, 500);
    chargedHadronEnergyFraction =
        ibooker.book1D("chargedHadronEnergyFraction", "chargedHadronEnergyFraction", 50, 0, 1);
    neutralHadronEnergyFraction =
        ibooker.book1D("neutralHadronEnergyFraction", "neutralHadronEnergyFraction", 50, 0, 1);
    chargedEmEnergyFraction = ibooker.book1D("chargedEmEnergyFraction", "chargedEmEnergyFraction", 50, 0, 1);
    neutralEmEnergyFraction = ibooker.book1D("neutralEmEnergyFraction", "neutralEmEnergyFraction", 50, 0, 1);
    photonEnergy = ibooker.book1D("photonEnergy", "photonEnergy", 50, 0, 500);
    photonEnergyFraction = ibooker.book1D("photonEnergyFraction", "photonEnergyFraction", 50, 0, 1);
    electronEnergy = ibooker.book1D("electronEnergy", "electronEnergy", 50, 0, 500);
    electronEnergyFraction = ibooker.book1D("electronEnergyFraction", "electronEnergyFraction", 50, 0, 1);
    muonEnergy = ibooker.book1D("muonEnergy", "muonEnergy", 50, 0, 500);
    muonEnergyFraction = ibooker.book1D("muonEnergyFraction", "muonEnergyFraction", 50, 0, 1);
    HFHadronEnergy = ibooker.book1D("HFHadronEnergy", "HFHadronEnergy", 50, 0, 500);
    HFHadronEnergyFraction = ibooker.book1D("HFHadronEnergyFraction", "HFHadronEnergyFraction", 50, 0, 1);
    HFEMEnergy = ibooker.book1D("HFEmEnergy", "HFEmEnergy", 50, 0, 500);
    HFEMEnergyFraction = ibooker.book1D("HFEmEnergyFraction", "HFEmEnergyFraction", 50, 0, 1);
    chargedHadronMultiplicity = ibooker.book1D("chargedHadronMultiplicity", "chargedHadronMultiplicity", 50, 0, 50);
    neutralHadronMultiplicity = ibooker.book1D("neutralHadronMultiplicity", "neutralHadronMultiplicity", 50, 0, 50);
    photonMultiplicity = ibooker.book1D("photonMultiplicity", "photonMultiplicity", 15, 0, 15);
    electronMultiplicity = ibooker.book1D("electronMultiplicity", "electronMultiplicity", 15, 0, 15);
    HFHadronMultiplicity = ibooker.book1D("HFHadronMultiplicity", "HFHadronMultiplicity", 50, 0, 50);
    HFEMMultiplicity = ibooker.book1D("HFEMMultiplicity", "HFEMMultiplicity", 50, 0, 50);
    chargedMuEnergy = ibooker.book1D("chargedMuEnergy", "chargedMuEnergy", 50, 0, 500);
    chargedMuEnergyFraction = ibooker.book1D("chargedMuEnergyFraction", "chargedMuEnergyFraction", 50, 0, 1);
    neutralMultiplicity = ibooker.book1D("neutralMultiplicity", "neutralMultiplicity", 50, 0, 50);
    HOEnergy = ibooker.book1D("HOEnergy", "HOEnergy", 50, 0, 500);
    HOEnergyFraction = ibooker.book1D("HOEnergyFraction", "HOEnergyFraction", 50, 0, 1);

    for (size_t j = 0; j < etaInfo.size(); ++j) {
      const auto& [etaRegion, etaLabel, etaMin, etaMax] = etaInfo[j];
      photonMultiplicity_EtaBins[j] = ibooker.book1D(
          std::format("photonMultiplicity_{}", etaRegion), std::format("photonMultiplicity_{}", etaRegion), 15, 0, 15);
      electronMultiplicity_EtaBins[j] = ibooker.book1D(std::format("electronMultiplicity_{}", etaRegion),
                                                       std::format("electronMultiplicity_{}", etaRegion),
                                                       15,
                                                       0,
                                                       15);
      neutralHadronMultiplicity_EtaBins[j] = ibooker.book1D(std::format("neutralHadronMultiplicity_{}", etaRegion),
                                                            std::format("neutralHadronMultiplicity_{}", etaRegion),
                                                            50,
                                                            0,
                                                            50);
      neutralMultiplicity_EtaBins[j] = ibooker.book1D(
          std::format("neutralMultiplicity_{}", etaRegion), std::format("neutralMultiplicity_{}", etaRegion), 50, 0, 50);
      chargedHadronMultiplicity_EtaBins[j] = ibooker.book1D(std::format("chargedHadronMultiplicity_{}", etaRegion),
                                                            std::format("chargedHadronMultiplicity_{}", etaRegion),
                                                            50,
                                                            0,
                                                            50);
      chargedMultiplicity_EtaBins[j] = ibooker.book1D(std::format("chargedMultiplicity_{}", etaRegion),
                                                      std::format("chargedMultiplicity_{}", etaRegion),
                                                      100,
                                                      0,
                                                      100);
    }
  }
}

//------------------------------------------------------------------------------
// ~JetTester
//------------------------------------------------------------------------------
JetTester::~JetTester() {}

//------------------------------------------------------------------------------
// analyze
//------------------------------------------------------------------------------
void JetTester::analyze(const edm::Event& mEvent, const edm::EventSetup& mSetup) {
  //------------------------------------------------------------------------------
  // Get the primary vertices
  //------------------------------------------------------------------------------

  edm::Handle<vector<reco::Vertex>> pvHandle;
  mEvent.getByToken(pvToken_, pvHandle);

  int nGoodVertices = 0;

  if (pvHandle.isValid()) {
    for (unsigned i = 0; i < pvHandle->size(); i++) {
      if ((*pvHandle)[i].ndof() > 4 && (std::abs((*pvHandle)[i].z()) <= 24) &&
          (std::abs((*pvHandle)[i].position().rho()) <= 2))
        nGoodVertices++;
    }
  }

  mNvtx->Fill(nGoodVertices);

  //------------------------------------------------------------------------------
  // Get the Jet collection
  //------------------------------------------------------------------------------

  bool correctionIsValid = false;
  edm::Handle<reco::JetCorrector> jetCorr;
  if (!isMiniAODJet && !mJetCorrector.label().empty()) {
    mEvent.getByToken(jetCorrectorToken_, jetCorr);
    if (jetCorr.isValid()) {
      correctionIsValid = true;
    }
  } else if (isMiniAODJet) {
    correctionIsValid = true;
  }

  std::vector<Jet> recoJets;
  std::vector<Jet> corrJets;
  std::vector<Jet> genJets;
  recoJets.clear();
  corrJets.clear();
  genJets.clear();

  edm::Handle<CaloJetCollection> caloJets;
  edm::Handle<PFJetCollection> pfJets;
  edm::Handle<pat::JetCollection> patJets;

  if (isCaloJet) {
    mEvent.getByToken(caloJetsToken_, caloJets);
    if (!caloJets.isValid())
      return;
    for (unsigned ijet = 0; ijet < caloJets->size(); ijet++) {
      recoJets.push_back((*caloJets)[ijet]);
      if (correctionIsValid) {
        auto jetCorrected = (*caloJets)[ijet];
        jetCorrected.scaleEnergy(jetCorr->correction(jetCorrected));
        corrJets.push_back(jetCorrected);
      }
    }
  } else if (isPFJet) {
    mEvent.getByToken(pfJetsToken_, pfJets);
    if (!pfJets.isValid())
      return;
    for (unsigned ijet = 0; ijet < pfJets->size(); ijet++) {
      // LEPTON CLEANING
      if (((*pfJets)[ijet].chargedMuEnergyFraction() > 0.8) || ((*pfJets)[ijet].electronEnergyFraction() > 0.8) ||
          ((*pfJets)[ijet].photonEnergyFraction() > 0.9))
        continue;
      recoJets.push_back((*pfJets)[ijet]);
      if (correctionIsValid) {
        auto jetCorrected = (*pfJets)[ijet];
        jetCorrected.scaleEnergy(jetCorr->correction(jetCorrected));
        corrJets.push_back(jetCorrected);
      }
    }
  } else if (isMiniAODJet) {
    mEvent.getByToken(patJetsToken_, patJets);
    if (!patJets.isValid())
      return;
    for (unsigned ijet = 0; ijet < patJets->size(); ijet++) {
      // LEPTON CLEANING
      if ((*patJets)[ijet].isPFJet() &&
          (((*patJets)[ijet].chargedMuEnergyFraction() > 0.8) || ((*patJets)[ijet].electronEnergyFraction() > 0.8) ||
           ((*patJets)[ijet].photonEnergyFraction() > 0.9)))
        continue;
      if (correctionIsValid) {
        corrJets.push_back((*patJets)[ijet]);
      }
      auto jet = (*patJets)[ijet];
      jet.scaleEnergy(jet.jecFactor("Uncorrected"));
      recoJets.push_back(jet);
    }
  }

  edm::Handle<GenJetCollection> genColl;
  if (!mEvent.isRealData()) {
    mEvent.getByToken(genJetsToken_, genColl);
    if (genColl.isValid() && !(mInputGenCollection.label().empty())) {
      for (unsigned gjet = 0; gjet < genColl->size(); gjet++) {
        genJets.push_back((*genColl)[gjet]);
      }
    }
  }

  int nJets = 0;
  std::vector<int> nJets_EtaBins(etaInfo.size(), 0);
  int index_first_jet = -1;
  double pt_first = -1;
  int index_second_jet = -1;
  double pt_second = -1;
  math::XYZTLorentzVector p4tmJetP[2];

  //------------------------------------------------------------------------------
  // Fill jet parameters
  //------------------------------------------------------------------------------

  for (unsigned ijet = 0; ijet < recoJets.size(); ijet++) {
    // PT CUT
    if (recoJets[ijet].pt() < mRecoJetPtThreshold)
      continue;

    nJets++;
    mJetEta->Fill(recoJets[ijet].eta());
    mJetPhi->Fill(recoJets[ijet].phi());
    mJetPt->Fill(recoJets[ijet].pt());
    mJetEtaPt->Fill(recoJets[ijet].eta(), recoJets[ijet].pt());
    mJetPhiPt->Fill(recoJets[ijet].phi(), recoJets[ijet].pt());
    mJetEnergy->Fill(recoJets[ijet].energy());
    mJetMass->Fill(recoJets[ijet].mass());
    mJetConstituents->Fill(recoJets[ijet].nConstituents());
    mJetArea->Fill(recoJets[ijet].jetArea());

    for (size_t j = 0; j < etaInfo.size(); ++j) {
      const auto& [etaRegion, etaLabel, etaMin, etaMax] = etaInfo[j];
      if (mInEtaBin(recoJets[ijet], etaMin, etaMax)) {
        nJets_EtaBins[j]++;
        mJetPt_EtaBins[j]->Fill(recoJets[ijet].pt());
      }
    }

    // Jet flavors contained in MiniAOD
    if (isMiniAODJet) {
      hadronFlavor->Fill((*patJets)[ijet].hadronFlavour());
      partonFlavor->Fill((*patJets)[ijet].partonFlavour());
      if ((*patJets)[ijet].genParton() != nullptr)
        genPartonPDGID->Fill((*patJets)[ijet].genParton()->pdgId());
    }

    if (!isMiniAODJet) {
      if (ijet == 0) {
        p4tmJetP[0] = recoJets[ijet].p4();
        mJetEtaFirst->Fill(recoJets[ijet].eta());
        mJetPhiFirst->Fill(recoJets[ijet].phi());
        mJetPtFirst->Fill(recoJets[ijet].pt());
      }
      if (ijet == 1) {
        p4tmJetP[1] = recoJets[ijet].p4();
      }
    } else {  // first jet might change after correction
      if ((recoJets[ijet].pt()) > pt_first) {
        pt_second = pt_first;
        pt_first = recoJets[ijet].pt();
        index_second_jet = index_first_jet;
        index_first_jet = ijet;
      } else if ((recoJets[ijet].pt()) > pt_second) {
        index_second_jet = ijet;
        pt_second = recoJets[ijet].pt();
      }
    }

    // ---- Calo Jet specific information ----
    if (isCaloJet) {
      maxEInEmTowers->Fill((*caloJets)[ijet].maxEInEmTowers());
      maxEInHadTowers->Fill((*caloJets)[ijet].maxEInHadTowers());
      energyFractionHadronic->Fill((*caloJets)[ijet].energyFractionHadronic());
      emEnergyFraction->Fill((*caloJets)[ijet].emEnergyFraction());
      hadEnergyInHB->Fill((*caloJets)[ijet].hadEnergyInHB());
      hadEnergyInHO->Fill((*caloJets)[ijet].hadEnergyInHO());
      hadEnergyInHE->Fill((*caloJets)[ijet].hadEnergyInHE());
      hadEnergyInHF->Fill((*caloJets)[ijet].hadEnergyInHF());
      emEnergyInEB->Fill((*caloJets)[ijet].emEnergyInEB());
      emEnergyInEE->Fill((*caloJets)[ijet].emEnergyInEE());
      emEnergyInHF->Fill((*caloJets)[ijet].emEnergyInHF());
      towersArea->Fill((*caloJets)[ijet].towersArea());
      n90->Fill((*caloJets)[ijet].n90());
      n60->Fill((*caloJets)[ijet].n60());
    }

    // ---- PF Jet specific information ----
    if (isPFJet) {
      muonMultiplicity->Fill((*pfJets)[ijet].muonMultiplicity());
      chargedMultiplicity->Fill((*pfJets)[ijet].chargedMultiplicity());
      chargedEmEnergy->Fill((*pfJets)[ijet].chargedEmEnergy());
      neutralEmEnergy->Fill((*pfJets)[ijet].neutralEmEnergy());
      chargedHadronEnergy->Fill((*pfJets)[ijet].chargedHadronEnergy());
      neutralHadronEnergy->Fill((*pfJets)[ijet].neutralHadronEnergy());
      chargedHadronEnergyFraction->Fill((*pfJets)[ijet].chargedHadronEnergyFraction());
      neutralHadronEnergyFraction->Fill((*pfJets)[ijet].neutralHadronEnergyFraction());
      chargedEmEnergyFraction->Fill((*pfJets)[ijet].chargedEmEnergyFraction());
      neutralEmEnergyFraction->Fill((*pfJets)[ijet].neutralEmEnergyFraction());
      photonEnergy->Fill((*pfJets)[ijet].photonEnergy());
      photonEnergyFraction->Fill((*pfJets)[ijet].photonEnergyFraction());
      electronEnergy->Fill((*pfJets)[ijet].electronEnergy());
      electronEnergyFraction->Fill((*pfJets)[ijet].electronEnergyFraction());
      muonEnergy->Fill((*pfJets)[ijet].muonEnergy());
      muonEnergyFraction->Fill((*pfJets)[ijet].muonEnergyFraction());
      HFHadronEnergy->Fill((*pfJets)[ijet].HFHadronEnergy());
      HFHadronEnergyFraction->Fill((*pfJets)[ijet].HFHadronEnergyFraction());
      HFEMEnergy->Fill((*pfJets)[ijet].HFEMEnergy());
      HFEMEnergyFraction->Fill((*pfJets)[ijet].HFEMEnergyFraction());
      chargedHadronMultiplicity->Fill((*pfJets)[ijet].chargedHadronMultiplicity());
      neutralHadronMultiplicity->Fill((*pfJets)[ijet].neutralHadronMultiplicity());
      photonMultiplicity->Fill((*pfJets)[ijet].photonMultiplicity());
      electronMultiplicity->Fill((*pfJets)[ijet].electronMultiplicity());
      HFHadronMultiplicity->Fill((*pfJets)[ijet].HFHadronMultiplicity());
      HFEMMultiplicity->Fill((*pfJets)[ijet].HFEMMultiplicity());
      chargedMuEnergy->Fill((*pfJets)[ijet].chargedMuEnergy());
      chargedMuEnergyFraction->Fill((*pfJets)[ijet].chargedMuEnergyFraction());
      neutralMultiplicity->Fill((*pfJets)[ijet].neutralMultiplicity());
      HOEnergy->Fill((*pfJets)[ijet].hoEnergy());
      HOEnergyFraction->Fill((*pfJets)[ijet].hoEnergyFraction());
      for (size_t j = 0; j < etaInfo.size(); ++j) {
        const auto& [etaRegion, etaLabel, etaMin, etaMax] = etaInfo[j];
        if (mInEtaBin(recoJets[ijet], etaMin, etaMax)) {
          photonMultiplicity_EtaBins[j]->Fill((*pfJets)[ijet].photonMultiplicity());
          electronMultiplicity_EtaBins[j]->Fill((*pfJets)[ijet].electronMultiplicity());
          neutralHadronMultiplicity_EtaBins[j]->Fill((*pfJets)[ijet].neutralHadronMultiplicity());
          neutralMultiplicity_EtaBins[j]->Fill((*pfJets)[ijet].neutralMultiplicity());
          chargedHadronMultiplicity_EtaBins[j]->Fill((*pfJets)[ijet].chargedHadronMultiplicity());
          chargedMultiplicity_EtaBins[j]->Fill((*pfJets)[ijet].chargedMultiplicity());

          h2d_chHadMult_vs_pt[j]->Fill(recoJets[ijet].pt(), (*pfJets)[ijet].chargedHadronMultiplicity());
          h2d_neHadMult_vs_pt[j]->Fill(recoJets[ijet].pt(), (*pfJets)[ijet].neutralHadronMultiplicity());
          h2d_chMult_vs_pt[j]->Fill(recoJets[ijet].pt(), (*pfJets)[ijet].chargedMultiplicity());
          h2d_neMult_vs_pt[j]->Fill(recoJets[ijet].pt(), (*pfJets)[ijet].neutralMultiplicity());
          h2d_phoMult_vs_pt[j]->Fill(recoJets[ijet].pt(), (*pfJets)[ijet].photonMultiplicity());

          h2d_chHad_vs_pt[j]->Fill(recoJets[ijet].pt(), (*pfJets)[ijet].chargedHadronEnergyFraction());
          h2d_neHad_vs_pt[j]->Fill(recoJets[ijet].pt(), (*pfJets)[ijet].neutralHadronEnergyFraction());
          h2d_chEm_vs_pt[j]->Fill(recoJets[ijet].pt(), (*pfJets)[ijet].chargedEmEnergyFraction());
          h2d_neEm_vs_pt[j]->Fill(recoJets[ijet].pt(), (*pfJets)[ijet].neutralEmEnergyFraction());
        }
      }
    }
    if (isMiniAODJet && (*patJets)[ijet].isPFJet()) {
      muonMultiplicity->Fill((*patJets)[ijet].muonMultiplicity());
      chargedMultiplicity->Fill((*patJets)[ijet].chargedMultiplicity());
      chargedEmEnergy->Fill((*patJets)[ijet].chargedEmEnergy());
      neutralEmEnergy->Fill((*patJets)[ijet].neutralEmEnergy());
      chargedHadronEnergy->Fill((*patJets)[ijet].chargedHadronEnergy());
      neutralHadronEnergy->Fill((*patJets)[ijet].neutralHadronEnergy());
      chargedHadronEnergyFraction->Fill((*patJets)[ijet].chargedHadronEnergyFraction());
      neutralHadronEnergyFraction->Fill((*patJets)[ijet].neutralHadronEnergyFraction());
      chargedEmEnergyFraction->Fill((*patJets)[ijet].chargedEmEnergyFraction());
      neutralEmEnergyFraction->Fill((*patJets)[ijet].neutralEmEnergyFraction());
      photonEnergy->Fill((*patJets)[ijet].photonEnergy());
      photonEnergyFraction->Fill((*patJets)[ijet].photonEnergyFraction());
      electronEnergy->Fill((*patJets)[ijet].electronEnergy());
      electronEnergyFraction->Fill((*patJets)[ijet].electronEnergyFraction());
      muonEnergy->Fill((*patJets)[ijet].muonEnergy());
      muonEnergyFraction->Fill((*patJets)[ijet].muonEnergyFraction());
      HFHadronEnergy->Fill((*patJets)[ijet].HFHadronEnergy());
      HFHadronEnergyFraction->Fill((*patJets)[ijet].HFHadronEnergyFraction());
      HFEMEnergy->Fill((*patJets)[ijet].HFEMEnergy());
      HFEMEnergyFraction->Fill((*patJets)[ijet].HFEMEnergyFraction());
      chargedHadronMultiplicity->Fill((*patJets)[ijet].chargedHadronMultiplicity());
      neutralHadronMultiplicity->Fill((*patJets)[ijet].neutralHadronMultiplicity());
      photonMultiplicity->Fill((*patJets)[ijet].photonMultiplicity());
      electronMultiplicity->Fill((*patJets)[ijet].electronMultiplicity());
      HFHadronMultiplicity->Fill((*patJets)[ijet].HFHadronMultiplicity());
      HFEMMultiplicity->Fill((*patJets)[ijet].HFEMMultiplicity());
      chargedMuEnergy->Fill((*patJets)[ijet].chargedMuEnergy());
      chargedMuEnergyFraction->Fill((*patJets)[ijet].chargedMuEnergyFraction());
      neutralMultiplicity->Fill((*patJets)[ijet].neutralMultiplicity());
      HOEnergy->Fill((*patJets)[ijet].hoEnergy());
      HOEnergyFraction->Fill((*patJets)[ijet].hoEnergyFraction());
    }

    //----------------------------------------------------------------------------
    // Match reco jets to gen jets for efficiency, fake and duplicate rates
    //----------------------------------------------------------------------------

    if (!mEvent.isRealData()) {
      double deltaRBestGen = 999;
      unsigned duplicateGenCounter = 0;
      for (unsigned gjet = 0; gjet < genJets.size(); gjet++) {
        // PT CUT
        if (genJets[gjet].pt() < mMatchGenPtThreshold)
          continue;
        double dR2 = deltaR2(genJets[gjet].eta(), genJets[gjet].phi(), recoJets[ijet].eta(), recoJets[ijet].phi());
        if (dR2 < mRThreshold * mRThreshold)
          duplicateGenCounter++;
        if (dR2 < deltaRBestGen * deltaRBestGen)
          deltaRBestGen = std::sqrt(dR2);
      }

      if (deltaRBestGen < mRThreshold) {
        mMatchedJetEta->Fill(recoJets[ijet].eta());
        mMatchedJetPhi->Fill(recoJets[ijet].phi());
        mMatchedJetPt->Fill(recoJets[ijet].pt());
        mMatchedJetEtaPt->Fill(recoJets[ijet].eta(), recoJets[ijet].pt());
        mMatchedJetPhiPt->Fill(recoJets[ijet].phi(), recoJets[ijet].pt());
        h2d_DeltaR_vs_Eta->Fill(recoJets[ijet].eta(), deltaRBestGen);
        h2d_DeltaR_vs_Phi->Fill(recoJets[ijet].phi(), deltaRBestGen);
        h2d_DeltaR_vs_Pt->Fill(recoJets[ijet].pt(), deltaRBestGen);

        if (isPFJet) {
          mMatchedJetchHad->Fill((*pfJets)[ijet].chargedHadronEnergyFraction());
          mMatchedJetneHad->Fill((*pfJets)[ijet].neutralHadronEnergyFraction());
          mMatchedJetchEm->Fill((*pfJets)[ijet].chargedEmEnergyFraction());
          mMatchedJetneEm->Fill((*pfJets)[ijet].neutralEmEnergyFraction());
          mMatchedJetnCost->Fill((*pfJets)[ijet].nConstituents());
        }

        for (size_t j = 0; j < etaInfo.size(); ++j) {
          const auto& [etaRegion, etaLabel, etaMin, etaMax] = etaInfo[j];
          if (mInEtaBin(recoJets[ijet], etaMin, etaMax)) {
            mMatchedJetPt_EtaBins[j]->Fill(recoJets[ijet].pt());
          }
        }
      }

      // measure gen duplicates: many gens to one reco
      if (duplicateGenCounter > 1) {
        mDuplicatesGenEta->Fill(recoJets[ijet].eta());
        mDuplicatesGenPhi->Fill(recoJets[ijet].phi());
        mDuplicatesGenPt->Fill(recoJets[ijet].pt());
        mDuplicatesGenEtaPt->Fill(recoJets[ijet].eta(), recoJets[ijet].pt());
        mDuplicatesGenPhiPt->Fill(recoJets[ijet].phi(), recoJets[ijet].pt());
        for (size_t j = 0; j < etaInfo.size(); ++j) {
          const auto& [etaRegion, etaLabel, etaMin, etaMax] = etaInfo[j];
          if (mInEtaBin(recoJets[ijet], etaMin, etaMax)) {
            mDuplicatesGenPt_EtaBins[j]->Fill(recoJets[ijet].pt());
          }
        }
      }
    }
  }

  if (!isMiniAODJet) {
    if (nJets >= 2) {
      mMjj->Fill((p4tmJetP[0] + p4tmJetP[1]).mass());
    }
  } else {
    if (index_first_jet > -1) {
      mJetEtaFirst->Fill(recoJets[index_first_jet].eta());
      mJetPhiFirst->Fill(recoJets[index_first_jet].phi());
      mJetPtFirst->Fill(recoJets[index_first_jet].pt() * (*patJets)[index_first_jet].jecFactor("Uncorrected"));
      p4tmJetP[0] = recoJets[index_first_jet].p4() * (*patJets)[index_first_jet].jecFactor("Uncorrected");
    }
    if (index_second_jet > -1) {
      p4tmJetP[1] = recoJets[index_second_jet].p4() * (*patJets)[index_second_jet].jecFactor("Uncorrected");
    }
    if ((index_first_jet > -1) && (index_second_jet > -1)) {
      mMjj->Fill((p4tmJetP[0] + p4tmJetP[1]).mass());
    }
  }

  //------------------------------------------------------------------------------
  // Count jets above pt cut
  //------------------------------------------------------------------------------

  mNJets->Fill(nJets);
  for (size_t j = 0; j < etaInfo.size(); ++j) {
    mNJets_EtaBins[j]->Fill(nJets_EtaBins[j]);
  }

  for (int istep = 0; istep < 100; ++istep) {
    int njets1 = 0;
    int njets2 = 0;

    float ptStep1 = (istep * (200. / 100.));
    float ptStep2 = (istep * (4000. / 100.));

    for (unsigned ijet = 0; ijet < recoJets.size(); ijet++) {
      if (!isMiniAODJet) {
        if (recoJets[ijet].pt() > ptStep1)
          njets1++;
        if (recoJets[ijet].pt() > ptStep2)
          njets2++;
      } else {
        if ((recoJets[ijet].pt() * (*patJets)[ijet].jecFactor("Uncorrected")) > ptStep1)
          njets1++;
        if ((recoJets[ijet].pt() * (*patJets)[ijet].jecFactor("Uncorrected")) > ptStep2)
          njets2++;
      }
      mNJetsPt1->Fill(ptStep1, njets1);
      mNJetsPt2->Fill(ptStep2, njets2);
    }
  }

  //------------------------------------------------------------------------------
  // Fill corrected jet parameters and corr vs reco
  //------------------------------------------------------------------------------

  if (correctionIsValid) {
    for (unsigned ijet = 0; ijet < recoJets.size(); ijet++) {
      // PT CUT
      if (corrJets[ijet].pt() < mRecoJetPtThreshold)
        continue;

      mCorrJetEta->Fill(corrJets[ijet].eta());
      mCorrJetPhi->Fill(corrJets[ijet].phi());
      mCorrJetPt->Fill(corrJets[ijet].pt());
      mCorrJetEtaPt->Fill(corrJets[ijet].eta(), corrJets[ijet].pt());
      mCorrJetPhiPt->Fill(corrJets[ijet].phi(), corrJets[ijet].pt());

      double ratio;
      if (recoJets[ijet].pt() != 0)
        ratio = corrJets[ijet].pt() / recoJets[ijet].pt();
      else
        continue;

      h2d_JetPtCorrOverReco_vs_Eta->Fill(corrJets[ijet].eta(), ratio);
      h2d_JetPtCorrOverReco_vs_Phi->Fill(corrJets[ijet].phi(), ratio);
      h2d_JetPtCorrOverReco_vs_Pt->Fill(corrJets[ijet].pt(), ratio);

      for (size_t j = 0; j < etaInfo.size(); ++j) {
        const auto& [etaRegion, etaLabel, etaMin, etaMax] = etaInfo[j];
        if (mInEtaBin(corrJets[ijet], etaMin, etaMax)) {
          mCorrJetPt_EtaBins[j]->Fill(corrJets[ijet].pt());
          h_JetPtCorrOverReco[j]->Fill(ratio);
          h2d_JetPtCorrOverReco_vs_Phi_EtaBins[j]->Fill(corrJets[ijet].phi(), ratio);
          h2d_JetPtCorrOverReco_vs_Pt_EtaBins[j]->Fill(corrJets[ijet].pt(), ratio);
          for (int i = 0; i < ptSize; ++i) {
            if ((recoJets[ijet].pt() > ptBins_[i]) && (recoJets[ijet].pt() <= ptBins_[i + 1]))
              hVector_JetPtCorrOverReco_ptBins[j][i]->Fill(ratio);
          }
        }
      }

      for (int i = 0; i < ptSize; ++i) {
        if ((recoJets[ijet].pt() > ptBins_[i]) && (recoJets[ijet].pt() <= ptBins_[i + 1])) {
          h2d_JetPtCorrOverReco_vs_Eta_ptBins[i]->Fill(corrJets[ijet].eta(), ratio);
        }
      }

      //----------------------------------------------------------------------------
      // Match reco jets to gen jets for efficiency
      //----------------------------------------------------------------------------
      if (!mEvent.isRealData()) {
        double deltaR2BestGen = 99999;
        for (unsigned gjet = 0; gjet < genJets.size(); gjet++) {
          double dR2 = deltaR2(genJets[gjet].eta(), genJets[gjet].phi(), corrJets[ijet].eta(), corrJets[ijet].phi());
          // PT CUT
          if (genJets[gjet].pt() < mMatchGenPtThreshold)
            continue;
          if (dR2 < deltaR2BestGen * deltaR2BestGen)
            deltaR2BestGen = dR2;
        }

        if (deltaR2BestGen < mRThreshold * mRThreshold) {
          if (corrJets[ijet].pt() > minJetPt) {
            for (size_t j = 0; j < etaInfo.size(); ++j) {
              const auto& [etaRegion, etaLabel, etaMin, etaMax] = etaInfo[j];
              if (mInEtaBin(corrJets[ijet], etaMin, etaMax)) {
                mMatchedCorrPt_EtaBins[j]->Fill(corrJets[ijet].pt());
              }
            }
          }
        }
      }  // if (!mEvent.isRealData()) {
    }
  }

  if (!mEvent.isRealData()) {
    //----------------------------------------------------------------------------
    // Fill Gen Jets histograms
    //----------------------------------------------------------------------------

    for (unsigned gjet = 0; gjet < genJets.size(); gjet++) {
      // MiniAOD has intrinsic thresholds, introduce threshold for RECO too
      // PT CUT
      if (genJets[gjet].pt() < mMatchGenPtThreshold)
        continue;
      if (std::abs(genJets[gjet].eta()) > 6.)
        continue;

      mGenEta->Fill(genJets[gjet].eta());
      mGenPhi->Fill(genJets[gjet].phi());
      mGenPt->Fill(genJets[gjet].pt());
      mGenEtaPt->Fill(genJets[gjet].eta(), genJets[gjet].pt());
      mGenPhiPt->Fill(genJets[gjet].phi(), genJets[gjet].pt());
      if (gjet == 0) {
        mGenEtaFirst->Fill(genJets[gjet].eta());
        mGenPhiFirst->Fill(genJets[gjet].phi());
        mGenPtFirst->Fill(genJets[gjet].pt());
      }

      for (size_t j = 0; j < etaInfo.size(); ++j) {
        const auto& [etaRegion, etaLabel, etaMin, etaMax] = etaInfo[j];
        if (mInEtaBin(genJets[gjet], etaMin, etaMax)) {
          mGenPt_EtaBins[j]->Fill(genJets[gjet].pt());
        }
      }

      if (recoJets.empty())
        continue;

      //----------------------------------------------------------------------------
      // Match gen jets to reco jets
      //----------------------------------------------------------------------------
      int iMatchReco = -1;
      unsigned duplicateRecoCounter = 0;
      double deltaRBestReco = 999;
      for (unsigned ijet = 0; ijet < recoJets.size(); ++ijet) {
        // PT CUT
        if (recoJets[ijet].pt() < mRecoJetPtThreshold)
          continue;
        double dR2 = deltaR2(genJets[gjet].eta(), genJets[gjet].phi(), recoJets[ijet].eta(), recoJets[ijet].phi());
        if (dR2 < mRThreshold * mRThreshold)
          duplicateRecoCounter++;
        if (dR2 < deltaRBestReco * deltaRBestReco) {
          iMatchReco = ijet;
          deltaRBestReco = std::sqrt(dR2);
        }
      }

      // measure reco duplicates: many recos to one gen
      if (duplicateRecoCounter > 1) {
        mDuplicatesJetEta->Fill(genJets[gjet].eta());
        mDuplicatesJetPhi->Fill(genJets[gjet].phi());
        mDuplicatesJetPt->Fill(genJets[gjet].pt());
        mDuplicatesJetEtaPt->Fill(genJets[gjet].eta(), genJets[gjet].pt());
        mDuplicatesJetPhiPt->Fill(genJets[gjet].phi(), genJets[gjet].pt());
        for (size_t j = 0; j < etaInfo.size(); ++j) {
          const auto& [etaRegion, etaLabel, etaMin, etaMax] = etaInfo[j];
          if (mInEtaBin(genJets[gjet], etaMin, etaMax)) {
            mDuplicatesJetPt_EtaBins[j]->Fill(genJets[gjet].pt());
          }
        }
      }

      if ((iMatchReco >= 0) && (deltaRBestReco < mRThreshold)) {
        //----------------------------------------------------------------------------
        // Fill gen jets to reco jets histograms
        //----------------------------------------------------------------------------

        mMatchedGenEta->Fill(genJets[gjet].eta());
        mMatchedGenPhi->Fill(genJets[gjet].phi());
        mMatchedGenPt->Fill(genJets[gjet].pt());
        mMatchedGenEtaPt->Fill(genJets[gjet].eta(), genJets[gjet].pt());
        mMatchedGenPhiPt->Fill(genJets[gjet].phi(), genJets[gjet].pt());
        h2d_DeltaR_vs_GenEta->Fill(genJets[gjet].eta(), deltaRBestReco);
        h2d_DeltaR_vs_GenPhi->Fill(genJets[gjet].phi(), deltaRBestReco);
        h2d_DeltaR_vs_GenPt->Fill(genJets[gjet].pt(), deltaRBestReco);

        double response;
        if (genJets[gjet].pt() != 0)
          response = (recoJets[iMatchReco].pt()) / genJets[gjet].pt();
        else
          continue;

        h2d_JetPtRecoOverGen_vs_GenEta->Fill(genJets[gjet].eta(), response);
        h2d_JetPtRecoOverGen_vs_GenPhi->Fill(genJets[gjet].phi(), response);
        h2d_JetPtRecoOverGen_vs_GenPt->Fill(genJets[gjet].pt(), response);

        for (size_t j = 0; j < etaInfo.size(); ++j) {
          const auto& [etaRegion, etaLabel, etaMin, etaMax] = etaInfo[j];
          if (mInEtaBin(genJets[gjet], etaMin, etaMax)) {
            mMatchedGenPt_EtaBins[j]->Fill(genJets[gjet].pt());
            h_JetPtRecoOverGen[j]->Fill(response);
            h2d_JetPtRecoOverGen_vs_GenPhi_EtaBins[j]->Fill(genJets[gjet].phi(), response);
            h2d_JetPtRecoOverGen_vs_GenPt_EtaBins[j]->Fill(genJets[gjet].pt(), response);
            if (isPFJet) {
              h2d_JetPtRecoOverGen_vs_chHad[j]->Fill((*pfJets)[iMatchReco].chargedHadronEnergyFraction(), response);
              h2d_JetPtRecoOverGen_vs_neHad[j]->Fill((*pfJets)[iMatchReco].neutralHadronEnergyFraction(), response);
              h2d_JetPtRecoOverGen_vs_chEm[j]->Fill((*pfJets)[iMatchReco].chargedEmEnergyFraction(), response);
              h2d_JetPtRecoOverGen_vs_neEm[j]->Fill((*pfJets)[iMatchReco].neutralEmEnergyFraction(), response);
              h2d_JetPtRecoOverGen_vs_nCost[j]->Fill(recoJets[iMatchReco].nConstituents(), response);
            }
            for (int i = 0; i < ptSize; ++i) {
              if ((genJets[gjet].pt() > ptBins_[i]) && (genJets[gjet].pt() <= ptBins_[i + 1])) {
                hVector_JetPtRecoOverGen_ptBins[j][i]->Fill(response);
              }
            }
          }
        }

        for (int i = 0; i < ptSize; ++i) {
          if ((genJets[gjet].pt() > ptBins_[i]) && (genJets[gjet].pt() <= ptBins_[i + 1])) {
            h2d_JetPtRecoOverGen_vs_GenEta_ptBins[i]->Fill(genJets[gjet].eta(), response);
          }
        }

        //----------------------------------------------------------------------------
        // Fill gen jets to corrected jets histograms
        //----------------------------------------------------------------------------

        if (correctionIsValid) {
          double responseCorr;
          if (genJets[gjet].pt() != 0)
            responseCorr = corrJets[iMatchReco].pt() / genJets[gjet].pt();
          else
            continue;

          h2d_JetPtCorrOverGen_vs_GenEta->Fill(genJets[gjet].eta(), responseCorr);
          h2d_JetPtCorrOverGen_vs_GenPhi->Fill(genJets[gjet].phi(), responseCorr);
          h2d_JetPtCorrOverGen_vs_GenPt->Fill(genJets[gjet].pt(), responseCorr);

          for (size_t j = 0; j < etaInfo.size(); ++j) {
            const auto& [etaRegion, etaLabel, etaMin, etaMax] = etaInfo[j];
            if (mInEtaBin(genJets[gjet], etaMin, etaMax)) {
              h_JetPtCorrOverGen[j]->Fill(responseCorr);
              h2d_JetPtCorrOverGen_vs_GenPt_EtaBins[j]->Fill(genJets[gjet].pt(), responseCorr);
              h2d_JetPtCorrOverGen_vs_GenPhi_EtaBins[j]->Fill(genJets[gjet].phi(), responseCorr);
              for (int i = 0; i < ptSize; ++i) {
                if ((genJets[gjet].pt() > ptBins_[i]) && (genJets[gjet].pt() <= ptBins_[i + 1]))
                  hVector_JetPtCorrOverGen_ptBins[j][i]->Fill(responseCorr);
              }
            }
          }

          for (int i = 0; i < ptSize; ++i) {
            if ((genJets[gjet].pt() > ptBins_[i]) && (genJets[gjet].pt() <= ptBins_[i + 1])) {
              h2d_JetPtCorrOverGen_vs_GenEta_ptBins[i]->Fill(genJets[gjet].eta(), responseCorr);
            }
          }

          mDeltaEta->Fill(genJets[gjet].eta() - recoJets[iMatchReco].eta());
          mDeltaPhi->Fill(genJets[gjet].phi() - recoJets[iMatchReco].phi());
          mDeltaPt->Fill((genJets[gjet].pt() - corrJets[iMatchReco].pt()) / genJets[gjet].pt());
        }
      }
    }
  }
}

//------------------------------------------------------------------------------
// fill description
//------------------------------------------------------------------------------
void JetTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  // Default jet validation offline
  desc.addUntracked<bool>("isHLT", false);
  desc.addUntracked<std::string>("JetType", "pf");
  desc.add<edm::InputTag>("src", edm::InputTag("ak4PFJets"));
  desc.add<edm::InputTag>("srcGen", edm::InputTag("ak4GenJetsNoNu"));
  desc.add<edm::InputTag>("JetCorrections", edm::InputTag("newAk4PFL1FastL2L3Corrector"));
  desc.add<edm::InputTag>("primVertex", edm::InputTag("offlinePrimaryVertices"));
  desc.add<double>("recoJetPtThreshold", 40.0);
  desc.add<double>("matchGenPtThreshold", 20.0);
  desc.add<double>("RThreshold", 0.3);
  descriptions.addWithDefaultLabel(desc);
}
