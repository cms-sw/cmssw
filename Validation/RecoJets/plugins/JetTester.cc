// Producer for validation histograms for Calo, JPT and PF jet objects
// F. Ratnikov, Sept. 7, 2006
// Modified by Chiyoung Jeong, Feb. 2, 2010
// Modified by J. Piedra, Sept. 11, 2013
// Modified by E. Vernazza, Aug. 1, 2025

#include "JetTester.h"

using namespace edm;
using namespace reco;
using namespace std;

JetTester::JetTester(const edm::ParameterSet &iConfig)
    : mInputCollection(iConfig.getParameter<edm::InputTag>("src")),
      //  rhoTag                         (iConfig.getParameter<edm::InputTag>
      //  ("srcRho")),
      JetType(iConfig.getUntrackedParameter<std::string>("JetType")),
      mRecoJetPtThreshold(iConfig.getParameter<double>("recoJetPtThreshold")),
      mMatchGenPtThreshold(iConfig.getParameter<double>("matchGenPtThreshold")),
      mRThreshold(iConfig.getParameter<double>("RThreshold")) {
  std::string inputCollectionLabel(mInputCollection.label());

  // Flag for the definition of Jet Input Collection 
  isCaloJet = (std::string("calo") == JetType); // <reco::CaloJetCollection>
  isPFJet = (std::string("pf") == JetType); // <reco::PFJetCollection>
  isMiniAODJet = (std::string("miniaod") == JetType); // <pat::JetCollection>
  if (!isCaloJet && !isPFJet && !isMiniAODJet) {
    throw cms::Exception("Configuration") << "Unknown jet type: " << JetType << "\nPlease use 'calo', 'pf', or 'miniaod'.";
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
  
  ptBins_ = {20., 30., 40., 100., 200., 300., 600., 2000., 5000., 6500., 1e6};
  n_bins_pt = ptBins_.size() - 1;

  // Events variables
  mNvtx = nullptr;
  
  // Jet parameters
  mJetEta = nullptr;
  mJetPhi = nullptr;
  mJetPt = nullptr;
  mJetEnergy = nullptr;
  mJetMass = nullptr;
  mJetConstituents = nullptr;
  mJetArea = nullptr;

  // Corrected jet parameters
  mCorrJetEta = nullptr;
  mCorrJetPhi = nullptr;
  mCorrJetPt = nullptr;
  
  // Gen jet parameters
  mGenEta = nullptr;
  mGenPhi = nullptr;
  mGenPt = nullptr;
  
  // Jet response vs gen histograms
  h_JetPtRecoOverGen_B = nullptr;
  h_JetPtRecoOverGen_E = nullptr;
  h_JetPtRecoOverGen_F = nullptr;
  hVector_JetPtRecoOverGen_B_ptBins.resize(n_bins_pt);
  hVector_JetPtRecoOverGen_E_ptBins.resize(n_bins_pt);
  hVector_JetPtRecoOverGen_F_ptBins.resize(n_bins_pt);

  // Corrected jet response vs gen histograms
  h_JetPtCorrOverGen_B = nullptr;
  h_JetPtCorrOverGen_E = nullptr;
  h_JetPtCorrOverGen_F = nullptr;
  hVector_JetPtCorrOverGen_B_ptBins.resize(n_bins_pt);
  hVector_JetPtCorrOverGen_E_ptBins.resize(n_bins_pt);
  hVector_JetPtCorrOverGen_F_ptBins.resize(n_bins_pt);

  // Corrected jet response vs reco histograms
  h_JetPtCorrOverReco_B = nullptr;
  h_JetPtCorrOverReco_E = nullptr;
  h_JetPtCorrOverReco_F = nullptr;
  hVector_JetPtCorrOverReco_B_ptBins.resize(n_bins_pt);
  hVector_JetPtCorrOverReco_E_ptBins.resize(n_bins_pt);
  hVector_JetPtCorrOverReco_F_ptBins.resize(n_bins_pt);
  
  // Jet response vs gen profiled in gen variable
  p_JetPtRecoOverGen_vs_GenEta = nullptr;
  p_JetPtRecoOverGen_vs_GenPhi_B = nullptr;
  p_JetPtRecoOverGen_vs_GenPhi_E = nullptr;
  p_JetPtRecoOverGen_vs_GenPhi_F = nullptr;
  p_JetPtRecoOverGen_vs_GenPt_B = nullptr;
  p_JetPtRecoOverGen_vs_GenPt_E = nullptr;
  p_JetPtRecoOverGen_vs_GenPt_F = nullptr;

  h2d_JetPtRecoOverGen_vs_GenEta = nullptr;
  h2d_JetPtRecoOverGen_vs_GenPhi_B = nullptr;
  h2d_JetPtRecoOverGen_vs_GenPhi_E = nullptr;
  h2d_JetPtRecoOverGen_vs_GenPhi_F = nullptr;
  h2d_JetPtRecoOverGen_vs_GenPt_B = nullptr;
  h2d_JetPtRecoOverGen_vs_GenPt_E = nullptr;
  h2d_JetPtRecoOverGen_vs_GenPt_F = nullptr;
  
  // Corrected jet response vs gen profiled in gen variable
  p_JetPtCorrOverGen_vs_GenEta = nullptr;
  p_JetPtCorrOverGen_vs_GenPhi_B = nullptr;
  p_JetPtCorrOverGen_vs_GenPhi_E = nullptr;
  p_JetPtCorrOverGen_vs_GenPhi_F = nullptr;
  p_JetPtCorrOverGen_vs_GenPt_B = nullptr;
  p_JetPtCorrOverGen_vs_GenPt_E = nullptr;
  p_JetPtCorrOverGen_vs_GenPt_F = nullptr;

  h2d_JetPtCorrOverGen_vs_GenEta = nullptr;
  h2d_JetPtCorrOverGen_vs_GenPhi_B = nullptr;
  h2d_JetPtCorrOverGen_vs_GenPhi_E = nullptr;
  h2d_JetPtCorrOverGen_vs_GenPhi_F = nullptr;
  h2d_JetPtCorrOverGen_vs_GenPt_B = nullptr;
  h2d_JetPtCorrOverGen_vs_GenPt_E = nullptr;
  h2d_JetPtCorrOverGen_vs_GenPt_F = nullptr;

  // Corrected jet response vs reco profiled in reco variable
  p_JetPtCorrOverReco_vs_Eta = nullptr;
  p_JetPtCorrOverReco_vs_Phi_B = nullptr;
  p_JetPtCorrOverReco_vs_Phi_E = nullptr;
  p_JetPtCorrOverReco_vs_Phi_F = nullptr;
  p_JetPtCorrOverReco_vs_Pt_B = nullptr;
  p_JetPtCorrOverReco_vs_Pt_E = nullptr;
  p_JetPtCorrOverReco_vs_Pt_F = nullptr;

  h2d_JetPtCorrOverReco_vs_Eta = nullptr;
  h2d_JetPtCorrOverReco_vs_Phi_B = nullptr;
  h2d_JetPtCorrOverReco_vs_Phi_E = nullptr;
  h2d_JetPtCorrOverReco_vs_Phi_F = nullptr;
  h2d_JetPtCorrOverReco_vs_Pt_B = nullptr;
  h2d_JetPtCorrOverReco_vs_Pt_E = nullptr;
  h2d_JetPtCorrOverReco_vs_Pt_F = nullptr;
  
  // First jet parameters
  mJetEtaFirst = nullptr;
  mJetPhiFirst = nullptr;
  mJetPtFirst = nullptr;
  mGenEtaFirst = nullptr;
  mGenPhiFirst = nullptr;
  mGenPtFirst = nullptr;
  
  // Other variables
  mMjj = nullptr;
  mNJets1 = nullptr;
  mNJets2 = nullptr;
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
  /// chargedHadronMultiplicity
  chargedHadronMultiplicity = nullptr;
  /// neutralHadronMultiplicity
  neutralHadronMultiplicity = nullptr;
  /// photonMultiplicity
  photonMultiplicity = nullptr;
  /// electronMultiplicity
  electronMultiplicity = nullptr;
  /// HFHadronMultiplicity
  HFHadronMultiplicity = nullptr;
  /// HFEMMultiplicity
  HFEMMultiplicity = nullptr;
  /// chargedMuEnergy
  chargedMuEnergy = nullptr;
  /// chargedMuEnergyFraction
  chargedMuEnergyFraction = nullptr;
  /// neutralMultiplicity
  neutralMultiplicity = nullptr;

  /// HOEnergy
  HOEnergy = nullptr;
  /// HOEnergyFraction (relative to corrected jet energy)
  HOEnergyFraction = nullptr;
  
  // contained in MiniAOD
  hadronFlavor = nullptr;
  partonFlavor = nullptr;
  genPartonPDGID = nullptr;

}

void JetTester::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const &iRun, edm::EventSetup const &) {
  ibooker.setCurrentFolder("JetMET/JetValidation/" + mInputCollection.label());

  // Discard all reco and corrected jets below this min threshold
  minJetPt = 20.;
  medJetPt = mRecoJetPtThreshold;

  int n_EtaBins = 60;
  int n_EtaBins_Profile = 20;
  std::vector<double> EtaRange = {-6.0, 6.0};
  int n_PhiBins = 70;
  int n_PhiBins_Profile = 20;
  std::vector<double> PhiRange = {-3.5, 3.5};
  int n_PtBins = 50;
  int n_PtBins_Profile = 100;
  std::vector<double> PtRange = {0, 1000};
  int n_RespBins = 60;
  std::vector<double> RespRange = {0, 3};

  // if eta range changed here need change in JetTesterPostProcessor as well
  // double etaBins[91] = {-6.0, -5.8, -5.6, -5.4, -5.2, -5.0, -4.8, -4.6, -4.4, -4.2, -4.0, -3.8, -3.6, -3.4, -3.2, -3.0,
  //                       -2.9, -2.8, -2.7, -2.6, -2.5, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4,
  //                       -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,  0.1,  0.2,
  //                       0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1.0,  1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,
  //                       1.9,  2.0,  2.1,  2.2,  2.3,  2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  3.0,  3.2,  3.4,  3.6,  3.8,
  //                       4.0,  4.2,  4.4,  4.6,  4.8,  5.0,  5.2,  5.4,  5.6,  5.8,  6.0};

  // Event variables
  mNvtx = ibooker.book1D("Nvtx", "number of vertices", 60, 0, 60);

  // Jet parameters
  mJetEta = ibooker.book1D("JetEta", "Reco Jets p_{T}>"+std::to_string(int(medJetPt))+" GeV;#eta;# jets", n_EtaBins, EtaRange[0], EtaRange[1]);
  mJetPhi = ibooker.book1D("JetPhi", "Reco Jets p_{T}>"+std::to_string(int(medJetPt))+" GeV;#phi;# jets", n_PhiBins, PhiRange[0], PhiRange[1]);
  mJetPt = ibooker.book1D("JetPt", "Reco Jets p_{T}>"+std::to_string(int(medJetPt))+" GeV;p_{T} [GeV];# jets", n_PtBins, PtRange[0], PtRange[1]);
  mJetEnergy = ibooker.book1D("JetEnergy", "Reco Jets;Energy [GeV];# jets", n_PtBins, PtRange[0], PtRange[1]);
  mJetMass = ibooker.book1D("JetMass", "Reco Jets;Mass [GeV];# jets", 100, 0, 200);
  mJetConstituents = ibooker.book1D("JetConstituents", "Reco Jets;# constituents;# jets", 100, 0, 100);
  mJetArea = ibooker.book1D("JetArea", "Reco Jets;Area;# jets", 100, 0, 4);

  // Gen jet parameters
  mGenEta = ibooker.book1D("GenEta", "Gen Jets p_{T}>"+std::to_string(int(mMatchGenPtThreshold))+" GeV;#eta;# jets", n_EtaBins, EtaRange[0], EtaRange[1]);
  mGenPhi = ibooker.book1D("GenPhi", "Gen Jets p_{T}>"+std::to_string(int(mMatchGenPtThreshold))+" GeV;#phi;# jets", n_PhiBins, PhiRange[0], PhiRange[1]);
  mGenPt = ibooker.book1D("GenPt", "Gen Jets p_{T}>"+std::to_string(int(mMatchGenPtThreshold))+" GeV;p_{T};# jets", n_PtBins, PtRange[0], PtRange[1]);

  // Jet response vs gen histograms
  h_JetPtRecoOverGen_B = ibooker.book1D("h_PtRecoOverGen_B", "Response Reco Jets - 0<|#eta|<1.5;p_{T}^{reco}/p_{T}^{gen};# jets", n_RespBins, RespRange[0], RespRange[1]);
  h_JetPtRecoOverGen_E = ibooker.book1D("h_PtRecoOverGen_E", "Response Reco Jets - 1.5<|#eta|<3;p_{T}^{reco}/p_{T}^{gen};# jets", n_RespBins, RespRange[0], RespRange[1]);
  h_JetPtRecoOverGen_F = ibooker.book1D("h_PtRecoOverGen_F", "Response Reco Jets - 3<|#eta|<6;p_{T}^{reco}/p_{T}^{gen};# jets", n_RespBins, RespRange[0], RespRange[1]);

  for (int i = 0; i < n_bins_pt; ++i) {
    double ptMin = ptBins_[i];
    double ptMax = ptBins_[i + 1];

    hVector_JetPtRecoOverGen_B_ptBins[i] = ibooker.book1D(
      "h_PtRecoOverGen_B_Pt"+std::to_string(int(ptMin))+"_"+std::to_string(int(ptMax)), 
      "Response Reco Jets - 0<|#eta|<1.5 - "+std::to_string(int(ptMin))+"<p_{T}^{gen}<"+std::to_string(int(ptMax))+";p_{T}^{reco}/p_{T}^{gen};# jets", 
      n_RespBins, RespRange[0], RespRange[1]);
    hVector_JetPtRecoOverGen_E_ptBins[i] = ibooker.book1D(
      "h_PtRecoOverGen_E_Pt"+std::to_string(int(ptMin))+"_"+std::to_string(int(ptMax)), 
      "Response Reco Jets - 1.5<|#eta|<3 - "+std::to_string(int(ptMin))+"<p_{T}^{gen}<"+std::to_string(int(ptMax))+";p_{T}^{reco}/p_{T}^{gen};# jets", 
      n_RespBins, RespRange[0], RespRange[1]);
    hVector_JetPtRecoOverGen_F_ptBins[i] = ibooker.book1D(
      "h_PtRecoOverGen_F_Pt"+std::to_string(int(ptMin))+"_"+std::to_string(int(ptMax)), 
      "Response Reco Jets - 3<|#eta|<6 - "+std::to_string(int(ptMin))+"<p_{T}^{gen}<"+std::to_string(int(ptMax))+";p_{T}^{reco}/p_{T}^{gen};# jets", 
      n_RespBins, RespRange[0], RespRange[1]);
  }

  // Jet response vs gen profiled in gen variable
  p_JetPtRecoOverGen_vs_GenEta = ibooker.bookProfile("pr_PtRecoOverGen_GenEta", 
    "Profiled Response Reco Jets;#eta^{gen};p_{T}^{reco}/p_{T}^{gen}",
    n_EtaBins_Profile, EtaRange[0], EtaRange[1], RespRange[0], RespRange[1], " ");
  p_JetPtRecoOverGen_vs_GenPhi_B = ibooker.bookProfile("pr_PtRecoOverGen_GenPhi_B", 
    "Profiled Response Reco Jets - 0<|#eta|<1.5;#phi^{gen};p_{T}^{reco}/p_{T}^{gen}",
    n_PhiBins_Profile, PhiRange[0], PhiRange[1], RespRange[0], RespRange[1], " ");
  p_JetPtRecoOverGen_vs_GenPhi_E = ibooker.bookProfile("pr_PtRecoOverGen_GenPhi_E", 
    "Profiled Response Reco Jets - 1.5<|#eta|<3;#phi^{gen};p_{T}^{reco}/p_{T}^{gen}",
    n_PhiBins_Profile, PhiRange[0], PhiRange[1], RespRange[0], RespRange[1], " ");
  p_JetPtRecoOverGen_vs_GenPhi_F = ibooker.bookProfile("pr_PtRecoOverGen_GenPhi_F", 
    "Profiled Response Reco Jets - 3<|#eta|<6;#phi^{gen};p_{T}^{reco}/p_{T}^{gen}",
    n_PhiBins_Profile, PhiRange[0], PhiRange[1], RespRange[0], RespRange[1], " ");
  p_JetPtRecoOverGen_vs_GenPt_B = ibooker.bookProfile("pr_PtRecoOverGen_GenPt_B", 
    "Profiled Response Reco Jets - 0<|#eta|<1.5;p_{T}^{gen};p_{T}^{reco}/p_{T}^{gen}",  
    n_PtBins_Profile, PtRange[0], PtRange[1], RespRange[0], RespRange[1], " ");
  p_JetPtRecoOverGen_vs_GenPt_E = ibooker.bookProfile("pr_PtRecoOverGen_GenPt_E", 
    "Profiled Response Reco Jets - 1.5<|#eta|<3;p_{T}^{gen};p_{T}^{reco}/p_{T}^{gen}",  
    n_PtBins_Profile, PtRange[0], PtRange[1], RespRange[0], RespRange[1], " ");
  p_JetPtRecoOverGen_vs_GenPt_F = ibooker.bookProfile("pr_PtRecoOverGen_GenPt_F", 
    "Profiled Response Reco Jets - 3<|#eta|<6;p_{T}^{gen};p_{T}^{reco}/p_{T}^{gen}",  
    n_PtBins_Profile, PtRange[0], PtRange[1], RespRange[0], RespRange[1], " ");

  h2d_JetPtRecoOverGen_vs_GenEta = ibooker.book2D("h2d_PtRecoOverGen_GenEta", 
    "Profiled Response Reco Jets;#eta^{gen};p_{T}^{reco}/p_{T}^{gen}",
    n_EtaBins_Profile, EtaRange[0], EtaRange[1], n_RespBins, RespRange[0], RespRange[1]);
  h2d_JetPtRecoOverGen_vs_GenPhi_B = ibooker.book2D("h2d_PtRecoOverGen_GenPhi_B", 
    "Profiled Response Reco Jets - 0<|#eta|<1.5;#phi^{gen};p_{T}^{reco}/p_{T}^{gen}",
    n_PhiBins_Profile, PhiRange[0], PhiRange[1], n_RespBins, RespRange[0], RespRange[1]);
  h2d_JetPtRecoOverGen_vs_GenPhi_E = ibooker.book2D("h2d_PtRecoOverGen_GenPhi_E", 
    "Profiled Response Reco Jets - 1.5<|#eta|<3;#phi^{gen};p_{T}^{reco}/p_{T}^{gen}",
    n_PhiBins_Profile, PhiRange[0], PhiRange[1], n_RespBins, RespRange[0], RespRange[1]);
  h2d_JetPtRecoOverGen_vs_GenPhi_F = ibooker.book2D("h2d_PtRecoOverGen_GenPhi_F", 
    "Profiled Response Reco Jets - 3<|#eta|<6;#phi^{gen};p_{T}^{reco}/p_{T}^{gen}",
    n_PhiBins_Profile, PhiRange[0], PhiRange[1], n_RespBins, RespRange[0], RespRange[1]);
  h2d_JetPtRecoOverGen_vs_GenPt_B = ibooker.book2D("h2d_PtRecoOverGen_GenPt_B", 
    "Profiled Response Reco Jets - 0<|#eta|<1.5;p_{T}^{gen};p_{T}^{reco}/p_{T}^{gen}",  
    n_PtBins_Profile, PtRange[0], PtRange[1], n_RespBins, RespRange[0], RespRange[1]);
  h2d_JetPtRecoOverGen_vs_GenPt_E = ibooker.book2D("h2d_PtRecoOverGen_GenPt_E", 
    "Profiled Response Reco Jets - 1.5<|#eta|<3;p_{T}^{gen};p_{T}^{reco}/p_{T}^{gen}",  
    n_PtBins_Profile, PtRange[0], PtRange[1], n_RespBins, RespRange[0], RespRange[1]);
  h2d_JetPtRecoOverGen_vs_GenPt_F = ibooker.book2D("h2d_PtRecoOverGen_GenPt_F", 
    "Profiled Response Reco Jets - 3<|#eta|<6;p_{T}^{gen};p_{T}^{reco}/p_{T}^{gen}",  
    n_PtBins_Profile, PtRange[0], PtRange[1], n_RespBins, RespRange[0], RespRange[1]);

  // Jet flavors contained in MiniAOD
  if (isMiniAODJet) {
    hadronFlavor = ibooker.book1D("HadronFlavor", ";Hadron Flavor;# jets", 44, -22, 22);
    partonFlavor = ibooker.book1D("PartonFlavor", ";Parton Flavor;# jets", 44, -22, 22);
    genPartonPDGID = ibooker.book1D("genPartonPDGID", ";genParton PDG ID;# jets", 44, -22, 22);
  }

  // Corrected jet parameters
  if (isMiniAODJet || !mJetCorrector.label().empty()) {  // if correction label is filled, but
                                                         // fill also for MiniAOD though
    mCorrJetEta = ibooker.book1D("CorrJetEta", "Corr Jets p_{T}>"+std::to_string(int(medJetPt))+" GeV;#eta;# jets", n_EtaBins, EtaRange[0], EtaRange[1]);
    mCorrJetPhi = ibooker.book1D("CorrJetPhi", "Corr Jets p_{T}>"+std::to_string(int(medJetPt))+" GeV;#phi;# jets", n_PhiBins, PhiRange[0], PhiRange[1]);
    mCorrJetPt = ibooker.book1D("CorrJetPt", "Corr Jets p_{T}>"+std::to_string(int(medJetPt))+" GeV;p_{T};# jets", n_PtBins, PtRange[0], PtRange[1]);

    // Corrected jet response vs gen histograms
    h_JetPtCorrOverGen_B = ibooker.book1D("h_PtCorrOverGen_B", "Response Corr Jets - 0<|#eta|<1.5;p_{T}^{corr}/p_{T}^{gen};# jets", n_RespBins, RespRange[0], RespRange[1]);
    h_JetPtCorrOverGen_E = ibooker.book1D("h_PtCorrOverGen_E", "Response Corr Jets - 1.5<|#eta|<3;p_{T}^{corr}/p_{T}^{gen};# jets", n_RespBins, RespRange[0], RespRange[1]);
    h_JetPtCorrOverGen_F = ibooker.book1D("h_PtCorrOverGen_F", "Response Corr Jets - 3<|#eta|<6;p_{T}^{corr}/p_{T}^{gen};# jets", n_RespBins, RespRange[0], RespRange[1]);

    // Corrected jet response vs reco histograms
    h_JetPtCorrOverReco_B = ibooker.book1D("h_PtCorrOverReco_B", "Response Corr Jets over Reco - 0<|#eta|<1.5;p_{T}^{corr}/p_{T}^{reco};# jets", n_RespBins, RespRange[0], RespRange[1]);
    h_JetPtCorrOverReco_E = ibooker.book1D("h_PtCorrOverReco_E", "Response Corr Jets over Reco - 1.5<|#eta|<3;p_{T}^{corr}/p_{T}^{reco};# jets", n_RespBins, RespRange[0], RespRange[1]);
    h_JetPtCorrOverReco_F = ibooker.book1D("h_PtCorrOverReco_F", "Response Corr Jets over Reco - 3<|#eta|<6;p_{T}^{corr}/p_{T}^{reco};# jets", n_RespBins, RespRange[0], RespRange[1]);

    for (int i = 0; i < n_bins_pt; ++i) {
      double ptMin = ptBins_[i];
      double ptMax = ptBins_[i + 1];

      hVector_JetPtCorrOverGen_B_ptBins[i] = ibooker.book1D(
        "h_PtCorrOverGen_B_Pt"+std::to_string(int(ptMin))+"_"+std::to_string(int(ptMax)), 
        "Response Corr Jets - 0<|#eta|<1.5 - "+std::to_string(int(ptMin))+"<p_{T}^{gen}<"+std::to_string(int(ptMax))+";p_{T}^{corr}/p_{T}^{gen};# jets", 
        n_RespBins, RespRange[0], RespRange[1]);
      hVector_JetPtCorrOverGen_E_ptBins[i] = ibooker.book1D(
        "h_PtCorrOverGen_E_Pt"+std::to_string(int(ptMin))+"_"+std::to_string(int(ptMax)), 
        "Response Corr Jets - 1.5<|#eta|<3 - "+std::to_string(int(ptMin))+"<p_{T}^{gen}<"+std::to_string(int(ptMax))+";p_{T}^{corr}/p_{T}^{gen};# jets", 
        n_RespBins, RespRange[0], RespRange[1]);
      hVector_JetPtCorrOverGen_F_ptBins[i] = ibooker.book1D(
        "h_PtCorrOverGen_F_Pt"+std::to_string(int(ptMin))+"_"+std::to_string(int(ptMax)), 
        "Response Corr Jets - 3<|#eta|<6 - "+std::to_string(int(ptMin))+"<p_{T}^{gen}<"+std::to_string(int(ptMax))+";p_{T}^{corr}/p_{T}^{gen};# jets", 
        n_RespBins, RespRange[0], RespRange[1]);

      hVector_JetPtCorrOverReco_B_ptBins[i] = ibooker.book1D(
        "h_PtCorrOverReco_B_Pt"+std::to_string(int(ptMin))+"_"+std::to_string(int(ptMax)), 
        "Response Corr Jets - 0<|#eta|<1.5 - "+std::to_string(int(ptMin))+"<p_{T}^{reco}<"+std::to_string(int(ptMax))+";p_{T}^{corr}/p_{T}^{reco};# jets", 
        n_RespBins, RespRange[0], RespRange[1]);
      hVector_JetPtCorrOverReco_E_ptBins[i] = ibooker.book1D(
        "h_PtCorrOverReco_E_Pt"+std::to_string(int(ptMin))+"_"+std::to_string(int(ptMax)), 
        "Response Corr Jets - 1.5<|#eta|<3 - "+std::to_string(int(ptMin))+"<p_{T}^{reco}<"+std::to_string(int(ptMax))+";p_{T}^{corr}/p_{T}^{reco};# jets", 
        n_RespBins, RespRange[0], RespRange[1]);
      hVector_JetPtCorrOverReco_F_ptBins[i] = ibooker.book1D(
        "h_PtCorrOverReco_F_Pt"+std::to_string(int(ptMin))+"_"+std::to_string(int(ptMax)), 
        "Response Corr Jets - 3<|#eta|<6 - "+std::to_string(int(ptMin))+"<p_{T}^{reco}<"+std::to_string(int(ptMax))+";p_{T}^{corr}/p_{T}^{reoc};# jets", 
        n_RespBins, RespRange[0], RespRange[1]);
    }

    // Corrected jet response vs gen profiled in gen variable
    p_JetPtCorrOverGen_vs_GenEta = ibooker.bookProfile("pr_PtCorrOverGen_GenEta", 
      "Profiled Response Corr Jets;#eta^{gen};p_{T}^{corr}/p_{T}^{gen}",
      n_EtaBins_Profile, EtaRange[0], EtaRange[1], RespRange[0], RespRange[1], " ");
    p_JetPtCorrOverGen_vs_GenPhi_B = ibooker.bookProfile("pr_PtCorrOverGen_GenPhi_B", 
      "Profiled Response Corr Jets - 0<|#eta|<1.5;#phi^{gen};p_{T}^{corr}/p_{T}^{gen}",
      n_PhiBins_Profile, PhiRange[0], PhiRange[1], RespRange[0], RespRange[1], " ");
    p_JetPtCorrOverGen_vs_GenPhi_E = ibooker.bookProfile("pr_PtCorrOverGen_GenPhi_E", 
      "Profiled Response Corr Jets - 1.5<|#eta|<3;#phi^{gen};p_{T}^{corr}/p_{T}^{gen}",
      n_PhiBins_Profile, PhiRange[0], PhiRange[1], RespRange[0], RespRange[1], " ");
    p_JetPtCorrOverGen_vs_GenPhi_F = ibooker.bookProfile("pr_PtCorrOverGen_GenPhi_F", 
      "Profiled Response Corr Jets - 3<|#eta|<6;#phi^{gen};p_{T}^{corr}/p_{T}^{gen}",
      n_PhiBins_Profile, PhiRange[0], PhiRange[1], RespRange[0], RespRange[1], " ");
    p_JetPtCorrOverGen_vs_GenPt_B = ibooker.bookProfile("pr_PtCorrOverGen_GenPt_B", 
      "Profiled Response Corr Jets - 0<|#eta|<1.5;p_{T}^{gen};p_{T}^{corr}/p_{T}^{gen}",  
      n_PtBins_Profile, PtRange[0], PtRange[1], RespRange[0], RespRange[1], " ");
    p_JetPtCorrOverGen_vs_GenPt_E = ibooker.bookProfile("pr_PtCorrOverGen_GenPt_E", 
      "Profiled Response Corr Jets - 1.5<|#eta|<3;p_{T}^{gen};p_{T}^{corr}/p_{T}^{gen}",  
      n_PtBins_Profile, PtRange[0], PtRange[1], RespRange[0], RespRange[1], " ");
    p_JetPtCorrOverGen_vs_GenPt_F = ibooker.bookProfile("pr_PtCorrOverGen_GenPt_F", 
      "Profiled Response Corr Jets - 3<|#eta|<6;p_{T}^{gen};p_{T}^{corr}/p_{T}^{gen}",  
      n_PtBins_Profile, PtRange[0], PtRange[1], RespRange[0], RespRange[1], " ");

    h2d_JetPtCorrOverGen_vs_GenEta = ibooker.book2D("h2d_PtCorrOverGen_GenEta", 
      "Profiled Response Corr Jets;#eta^{gen};p_{T}^{corr}/p_{T}^{gen}",
      n_EtaBins_Profile, EtaRange[0], EtaRange[1], n_RespBins, RespRange[0], RespRange[1]);
    h2d_JetPtCorrOverGen_vs_GenPhi_B = ibooker.book2D("h2d_PtCorrOverGen_GenPhi_B", 
      "Profiled Response Corr Jets - 0<|#eta|<1.5;#phi^{gen};p_{T}^{corr}/p_{T}^{gen}",
      n_PhiBins_Profile, PhiRange[0], PhiRange[1], n_RespBins, RespRange[0], RespRange[1]);
    h2d_JetPtCorrOverGen_vs_GenPhi_E = ibooker.book2D("h2d_PtCorrOverGen_GenPhi_E", 
      "Profiled Response Corr Jets - 1.5<|#eta|<3;#phi^{gen};p_{T}^{corr}/p_{T}^{gen}",
      n_PhiBins_Profile, PhiRange[0], PhiRange[1], n_RespBins, RespRange[0], RespRange[1]);
    h2d_JetPtCorrOverGen_vs_GenPhi_F = ibooker.book2D("h2d_PtCorrOverGen_GenPhi_F", 
      "Profiled Response Corr Jets - 3<|#eta|<6;#phi^{gen};p_{T}^{corr}/p_{T}^{gen}",
      n_PhiBins_Profile, PhiRange[0], PhiRange[1], n_RespBins, RespRange[0], RespRange[1]);
    h2d_JetPtCorrOverGen_vs_GenPt_B = ibooker.book2D("h2d_PtCorrOverGen_GenPt_B", 
      "Profiled Response Corr Jets - 0<|#eta|<1.5;p_{T}^{gen};p_{T}^{corr}/p_{T}^{gen}",  
      n_PtBins_Profile, PtRange[0], PtRange[1], n_RespBins, RespRange[0], RespRange[1]);
    h2d_JetPtCorrOverGen_vs_GenPt_E = ibooker.book2D("h2d_PtCorrOverGen_GenPt_E", 
      "Profiled Response Corr Jets - 1.5<|#eta|<3;p_{T}^{gen};p_{T}^{corr}/p_{T}^{gen}",  
      n_PtBins_Profile, PtRange[0], PtRange[1], n_RespBins, RespRange[0], RespRange[1]);
    h2d_JetPtCorrOverGen_vs_GenPt_F = ibooker.book2D("h2d_PtCorrOverGen_GenPt_F", 
      "Profiled Response Corr Jets - 3<|#eta|<6;p_{T}^{gen};p_{T}^{corr}/p_{T}^{gen}",  
      n_PtBins_Profile, PtRange[0], PtRange[1], n_RespBins, RespRange[0], RespRange[1]);
    
    // Corrected jet response vs reco profiled in reco variable
    p_JetPtCorrOverReco_vs_Eta = ibooker.bookProfile("pr_PtCorrOverReco_Eta", 
      "Profiled Response Corr Jets;#eta^{reco};p_{T}^{corr}/p_{T}^{reco}",
      n_EtaBins_Profile, EtaRange[0], EtaRange[1], RespRange[0], RespRange[1], " ");
    p_JetPtCorrOverReco_vs_Phi_B = ibooker.bookProfile("pr_PtCorrOverReco_Phi_B", 
      "Profiled Response Corr Jets - 0<|#eta|<1.5;#phi^{reco};p_{T}^{corr}/p_{T}^{reco}",
      n_PhiBins_Profile, PhiRange[0], PhiRange[1], RespRange[0], RespRange[1], " ");
    p_JetPtCorrOverReco_vs_Phi_E = ibooker.bookProfile("pr_PtCorrOverReco_Phi_E", 
      "Profiled Response Corr Jets - 1.5<|#eta|<3;#phi^{reco};p_{T}^{corr}/p_{T}^{reco}",
      n_PhiBins_Profile, PhiRange[0], PhiRange[1], RespRange[0], RespRange[1], " ");
    p_JetPtCorrOverReco_vs_Phi_F = ibooker.bookProfile("pr_PtCorrOverReco_Phi_F", 
      "Profiled Response Corr Jets - 3<|#eta|<6;#phi^{reco};p_{T}^{corr}/p_{T}^{reco}",
      n_PhiBins_Profile, PhiRange[0], PhiRange[1], RespRange[0], RespRange[1], " ");
    p_JetPtCorrOverReco_vs_Pt_B = ibooker.bookProfile("pr_PtCorrOverReco_Pt_B", 
      "Profiled Response Corr Jets - 0<|#eta|<1.5;p_{T}^{reco};p_{T}^{corr}/p_{T}^{reco}",  
      n_PtBins_Profile, PtRange[0], PtRange[1], RespRange[0], RespRange[1], " ");
    p_JetPtCorrOverReco_vs_Pt_E = ibooker.bookProfile("pr_PtCorrOverReco_Pt_E", 
      "Profiled Response Corr Jets - 1.5<|#eta|<3;p_{T}^{reco};p_{T}^{corr}/p_{T}^{reco}",  
      n_PtBins_Profile, PtRange[0], PtRange[1], RespRange[0], RespRange[1], " ");
    p_JetPtCorrOverReco_vs_Pt_F = ibooker.bookProfile("pr_PtCorrOverReco_Pt_F", 
      "Profiled Response Corr Jets - 3<|#eta|<6;p_{T}^{reco};p_{T}^{corr}/p_{T}^{reco}",  
      n_PtBins_Profile, PtRange[0], PtRange[1], RespRange[0], RespRange[1], " ");

    h2d_JetPtCorrOverReco_vs_Eta = ibooker.book2D("h2d_PtCorrOverReco_Eta", 
      "Profiled Response Corr Jets;#eta^{reco};p_{T}^{corr}/p_{T}^{reco}",
      n_EtaBins_Profile, EtaRange[0], EtaRange[1], n_RespBins, RespRange[0], RespRange[1]);
    h2d_JetPtCorrOverReco_vs_Phi_B = ibooker.book2D("h2d_PtCorrOverReco_Phi_B", 
      "Profiled Response Corr Jets - 0<|#eta|<1.5;#phi^{reco};p_{T}^{corr}/p_{T}^{reco}",
      n_PhiBins_Profile, PhiRange[0], PhiRange[1], n_RespBins, RespRange[0], RespRange[1]);
    h2d_JetPtCorrOverReco_vs_Phi_E = ibooker.book2D("h2d_PtCorrOverReco_Phi_E", 
      "Profiled Response Corr Jets - 1.5<|#eta|<3;#phi^{reco};p_{T}^{corr}/p_{T}^{reco}",
      n_PhiBins_Profile, PhiRange[0], PhiRange[1], n_RespBins, RespRange[0], RespRange[1]);
    h2d_JetPtCorrOverReco_vs_Phi_F = ibooker.book2D("h2d_PtCorrOverReco_Phi_F", 
      "Profiled Response Corr Jets - 3<|#eta|<6;#phi^{reco};p_{T}^{corr}/p_{T}^{reco}",
      n_PhiBins_Profile, PhiRange[0], PhiRange[1], n_RespBins, RespRange[0], RespRange[1]);
    h2d_JetPtCorrOverReco_vs_Pt_B = ibooker.book2D("h2d_PtCorrOverReco_Pt_B", 
      "Profiled Response Corr Jets - 0<|#eta|<1.5;p_{T}^{reco};p_{T}^{corr}/p_{T}^{reco}",  
      n_PtBins_Profile, PtRange[0], PtRange[1], n_RespBins, RespRange[0], RespRange[1]);
    h2d_JetPtCorrOverReco_vs_Pt_E = ibooker.book2D("h2d_PtCorrOverReco_Pt_E", 
      "Profiled Response Corr Jets - 1.5<|#eta|<3;p_{T}^{reco};p_{T}^{corr}/p_{T}^{reco}",  
      n_PtBins_Profile, PtRange[0], PtRange[1], n_RespBins, RespRange[0], RespRange[1]);
    h2d_JetPtCorrOverReco_vs_Pt_F = ibooker.book2D("h2d_PtCorrOverReco_Pt_F", 
      "Profiled Response Corr Jets - 3<|#eta|<6;p_{T}^{reco};p_{T}^{corr}/p_{T}^{reco}",  
      n_PtBins_Profile, PtRange[0], PtRange[1], n_RespBins, RespRange[0], RespRange[1]);

  }

  // Generation
  mJetEtaFirst = ibooker.book1D("FirstJetEta", "First Jets;#eta;# first jets", n_EtaBins, EtaRange[0], EtaRange[1]);
  mJetPhiFirst = ibooker.book1D("FirstJetPhi", "First Jets;#phi;# first jets", n_PhiBins, PhiRange[0], PhiRange[1]);
  mJetPtFirst = ibooker.book1D("FirstJetPt", "First Jets;p_{T};# first jets", n_PtBins, PtRange[0], PtRange[1]);
  mGenEtaFirst = ibooker.book1D("FirstGenJetEta", "First Gen Jets;#eta;# first jets", n_EtaBins, EtaRange[0], EtaRange[1]);
  mGenPhiFirst = ibooker.book1D("FirstGenJetPhi", "First Gen Jets;#phi;# first jets", n_PhiBins, PhiRange[0], PhiRange[1]);
  mGenPtFirst = ibooker.book1D("FirstGenJetPt", "First Gen Jets;p_{T};# first jets", n_PtBins, PtRange[0], PtRange[1]);
  
  // Some jet algebra
  mMjj = ibooker.book1D("Mjj", "Mjj", 100, 0, 2000);
  mNJets1 = ibooker.bookProfile("NJets1", "Number of jets above Pt threshold", 100, 0, 200, 100, 0, 50, "s");
  mNJets2 = ibooker.bookProfile("NJets2", "Number of jets above Pt threshold", 100, 0, 4000, 100, 0, 50, "s");
  mDeltaEta = ibooker.book1D("DeltaEta", ";#eta^{gen}-#eta^{corr};# matched jets", 100, -0.5, 0.5);
  mDeltaPhi = ibooker.book1D("DeltaPhi", ";#phi^{gen}-#phi^{corr};# matched jets", 100, -0.5, 0.5);
  mDeltaPt = ibooker.book1D("DeltaPt", ";(p_{T}^{gen}-p_{T}^{corr})/p_{T}^{gen};# matched jets", 100, -1.0, 1.0);
  
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
    photonMultiplicity = ibooker.book1D("photonMultiplicity", "photonMultiplicity", 10, 0, 10);
    electronMultiplicity = ibooker.book1D("electronMultiplicity", "electronMultiplicity", 10, 0, 10);
    HFHadronMultiplicity = ibooker.book1D("HFHadronMultiplicity", "HFHadronMultiplicity", 50, 0, 50);
    HFEMMultiplicity = ibooker.book1D("HFEMMultiplicity", "HFEMMultiplicity", 50, 0, 50);
    chargedMuEnergy = ibooker.book1D("chargedMuEnergy", "chargedMuEnergy", 50, 0, 500);
    chargedMuEnergyFraction = ibooker.book1D("chargedMuEnergyFraction", "chargedMuEnergyFraction", 50, 0, 1);
    neutralMultiplicity = ibooker.book1D("neutralMultiplicity", "neutralMultiplicity", 50, 0, 50);
    HOEnergy = ibooker.book1D("HOEnergy", "HOEnergy", 50, 0, 500);
    HOEnergyFraction = ibooker.book1D("HOEnergyFraction", "HOEnergyFraction", 50, 0, 1);
  }
}

//------------------------------------------------------------------------------
// ~JetTester
//------------------------------------------------------------------------------
JetTester::~JetTester() {}

//------------------------------------------------------------------------------
// analyze
//------------------------------------------------------------------------------
void JetTester::analyze(const edm::Event &mEvent, const edm::EventSetup &mSetup) {

  //------------------------------------------------------------------------------
  // Get the primary vertices
  //------------------------------------------------------------------------------
  edm::Handle<vector<reco::Vertex>> pvHandle;
  mEvent.getByToken(pvToken_, pvHandle);

  int nGoodVertices = 0;

  if (pvHandle.isValid()) {
    for (unsigned i = 0; i < pvHandle->size(); i++) {
      if ((*pvHandle)[i].ndof() > 4 && (std::abs((*pvHandle)[i].z()) <= 24) && (std::abs((*pvHandle)[i].position().rho()) <= 2))
        nGoodVertices++;
    }
  }

  mNvtx->Fill(nGoodVertices);

  //------------------------------------------------------------------------------
  // Get the Jet collection
  //------------------------------------------------------------------------------

  math::XYZTLorentzVector p4tmJetP[2];

  std::vector<Jet> recoJets;
  recoJets.clear();

  edm::Handle<CaloJetCollection> caloJets;
  edm::Handle<PFJetCollection> pfJets;
  //  edm::Handle<JPTJetCollection>  jptJets;
  edm::Handle<pat::JetCollection> patJets;

  if (isCaloJet)
    mEvent.getByToken(caloJetsToken_, caloJets);
  if (isPFJet)
    mEvent.getByToken(pfJetsToken_, pfJets);
  //  if (isJPTJet)  mEvent.getByToken(jptJetsToken_, jptJets);
  if (isMiniAODJet)
    mEvent.getByToken(patJetsToken_, patJets);

  if (isCaloJet && !caloJets.isValid())
    return;
  if (isPFJet && !pfJets.isValid())
    return;
  //  if (isJPTJet  && !jptJets.isValid())  return;
  if (isMiniAODJet && !patJets.isValid())
    return;

  if (isCaloJet) {
    for (unsigned ijet = 0; ijet < caloJets->size(); ijet++)
      recoJets.push_back((*caloJets)[ijet]);
  }
  /*  if (isJPTJet)
      {
        for (unsigned ijet=0; ijet<jptJets->size(); ijet++)
          recoJets.push_back((*jptJets)[ijet]);
      }*/
  if (isPFJet) {
    for (unsigned ijet = 0; ijet < pfJets->size(); ijet++)
      recoJets.push_back((*pfJets)[ijet]);
  }
  if (isMiniAODJet) {
    for (unsigned ijet = 0; ijet < patJets->size(); ijet++)
      recoJets.push_back((*patJets)[ijet]);
  }

  int nJet = 0;
  int index_first_jet = -1;
  double pt_first = -1;
  int index_second_jet = -1;
  double pt_second = -1;

  //------------------------------------------------------------------------------
  // Fill jet parameters for pass_mediumjet
  //------------------------------------------------------------------------------

  for (unsigned ijet = 0; ijet < recoJets.size(); ijet++) {

    // Define correction factor for MiniAODJet (otherwise 1)
    double jec_factor = 1.;
    if (isMiniAODJet) {
      jec_factor = (*patJets)[ijet].jecFactor("Uncorrected");
    }

    bool pass_mediumjet = false;
    if (recoJets[ijet].pt() * jec_factor > medJetPt) {
      pass_mediumjet = true;
    }
    if (pass_mediumjet) {

      if (mJetEta)
        mJetEta->Fill(recoJets[ijet].eta());
      if (mJetPhi)
        mJetPhi->Fill(recoJets[ijet].phi());
      if (mJetPt)
        mJetPt->Fill(recoJets[ijet].pt() * jec_factor);
      if (mJetEnergy)
        mJetEnergy->Fill(recoJets[ijet].energy() * jec_factor);
      if (mJetMass)
        mJetMass->Fill(recoJets[ijet].mass() * jec_factor);
      if (mJetConstituents)
        mJetConstituents->Fill(recoJets[ijet].nConstituents());
      if (mJetArea)
        mJetArea->Fill(recoJets[ijet].jetArea());

      // Jet flavors contained in MiniAOD
      if (isMiniAODJet) {
        if (hadronFlavor)
          hadronFlavor->Fill((*patJets)[ijet].hadronFlavour());
        if (partonFlavor)
          partonFlavor->Fill((*patJets)[ijet].partonFlavour());
        if (genPartonPDGID && (*patJets)[ijet].genParton() != nullptr)
          genPartonPDGID->Fill((*patJets)[ijet].genParton()->pdgId());
      }

      if (!isMiniAODJet) {
        if (ijet == 0) {
          nJet++;
          p4tmJetP[0] = recoJets[ijet].p4();
          if (mJetEtaFirst)
            mJetEtaFirst->Fill(recoJets[ijet].eta());
          if (mJetPhiFirst)
            mJetPhiFirst->Fill(recoJets[ijet].phi());
          if (mJetPtFirst)
            mJetPtFirst->Fill(recoJets[ijet].pt());
        }
        if (ijet == 1) {
          nJet++;
          p4tmJetP[1] = recoJets[ijet].p4();
        }
      } else { // first jet might change after correction
        if ((recoJets[ijet].pt() * jec_factor) > pt_first) {
          pt_second = pt_first;
          pt_first = recoJets[ijet].pt() * jec_factor;
          index_second_jet = index_first_jet;
          index_first_jet = ijet;
        } else if ((recoJets[ijet].pt() * jec_factor) > pt_second) {
          index_second_jet = ijet;
          pt_second = recoJets[ijet].pt() * jec_factor;
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
    }
  }

  if (!isMiniAODJet) {
    if (nJet >= 2) {
      if (mMjj)
        mMjj->Fill((p4tmJetP[0] + p4tmJetP[1]).mass());
    }
  } else {
    if (index_first_jet > -1) {
      if (mJetEtaFirst)
        mJetEtaFirst->Fill(recoJets[index_first_jet].eta());
      if (mJetPhiFirst)
        mJetPhiFirst->Fill(recoJets[index_first_jet].phi());
      if (mJetPtFirst)
        mJetPtFirst->Fill(recoJets[index_first_jet].pt() * (*patJets)[index_first_jet].jecFactor("Uncorrected"));
      nJet++;
      p4tmJetP[0] = recoJets[index_first_jet].p4() * (*patJets)[index_first_jet].jecFactor("Uncorrected");
    }
    if (index_second_jet > -1) {
      nJet++;
      p4tmJetP[1] = recoJets[index_second_jet].p4() * (*patJets)[index_second_jet].jecFactor("Uncorrected");
    }
    if (nJet >= 2) {
      if (mMjj)
        mMjj->Fill((p4tmJetP[0] + p4tmJetP[1]).mass());
    }
  }

  //------------------------------------------------------------------------------
  // Count jets above pt cut
  //------------------------------------------------------------------------------

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
      mNJets1->Fill(ptStep1, njets1);
      mNJets2->Fill(ptStep2, njets2);
    }
  }

  //------------------------------------------------------------------------------
  // Fill corrected jet parameters and corr vs reco
  //------------------------------------------------------------------------------

  double scale = -999;
  edm::Handle<reco::JetCorrector> jetCorr;
  bool pass_correction_flag = false;
  if (!isMiniAODJet && !mJetCorrector.label().empty()) {
    mEvent.getByToken(jetCorrectorToken_, jetCorr);
    if (jetCorr.isValid()) {
      pass_correction_flag = true;
    }
  } else if (isMiniAODJet) {
    pass_correction_flag = true;
  }

  for (unsigned ijet = 0; ijet < recoJets.size(); ijet++) {

    Jet correctedJet = recoJets[ijet];
    if (pass_correction_flag) {
      if (isCaloJet)
        scale = jetCorr->correction((*caloJets)[ijet]);
      if (isPFJet)
        scale = jetCorr->correction((*pfJets)[ijet]);
      // if (isJPTJet)  scale = jetCorr->correction((*jptJets)[ijet]);
      if (!isMiniAODJet) {
        correctedJet.scaleEnergy(scale);
      }

      if (correctedJet.pt() < minJetPt) continue;

      if (correctedJet.pt() > medJetPt) {
        mCorrJetEta->Fill(correctedJet.eta());
        mCorrJetPhi->Fill(correctedJet.phi());
        mCorrJetPt->Fill(correctedJet.pt());
      }

      double ijetEta = recoJets[ijet].eta();
      double ijetPhi = recoJets[ijet].phi();
      double ijetPt = recoJets[ijet].pt();
      double ratio = correctedJet.pt() / ijetPt;
      if (isMiniAODJet) {
        ijetPt = recoJets[ijet].pt() * (*patJets)[ijet].jecFactor("Uncorrected");
        ratio = 1. / (*patJets)[ijet].jecFactor("Uncorrected");
      }

      p_JetPtCorrOverReco_vs_Eta->Fill(ijetEta, ratio);
      h2d_JetPtCorrOverReco_vs_Eta->Fill(ijetEta, ratio);
      if (std::abs(ijetEta) < 1.5) {
        h_JetPtCorrOverReco_B->Fill(ratio);
        p_JetPtCorrOverReco_vs_Phi_B->Fill(ijetPhi, ratio);
        h2d_JetPtCorrOverReco_vs_Phi_B->Fill(ijetPhi, ratio);
        p_JetPtCorrOverReco_vs_Pt_B->Fill(ijetPt, ratio);
        h2d_JetPtCorrOverReco_vs_Pt_B->Fill(ijetPt, ratio);
        for (int i = 0; i < n_bins_pt; ++i) {
          if ((ijetPt > ptBins_[i]) && (ijetPt < ptBins_[i + 1]))
            hVector_JetPtCorrOverReco_B_ptBins[i]->Fill(ratio);
        }
      } else if (std::abs(ijetEta) < 3.0) {
        h_JetPtCorrOverReco_E->Fill(ratio);
        p_JetPtCorrOverReco_vs_Phi_E->Fill(ijetPhi, ratio);
        h2d_JetPtCorrOverReco_vs_Phi_E->Fill(ijetPhi, ratio);
        p_JetPtCorrOverReco_vs_Pt_E->Fill(ijetPt, ratio);
        h2d_JetPtCorrOverReco_vs_Pt_E->Fill(ijetPt, ratio);
        for (int i = 0; i < n_bins_pt; ++i) {
          if ((ijetPt > ptBins_[i]) && (ijetPt < ptBins_[i + 1]))
            hVector_JetPtCorrOverReco_E_ptBins[i]->Fill(ratio);
        }
      } else if (std::abs(ijetEta) < 6.0) {
        h_JetPtCorrOverReco_F->Fill(ratio);
        p_JetPtCorrOverReco_vs_Phi_F->Fill(ijetPhi, ratio);
        h2d_JetPtCorrOverReco_vs_Phi_F->Fill(ijetPhi, ratio);
        p_JetPtCorrOverReco_vs_Pt_F->Fill(ijetPt, ratio);
        h2d_JetPtCorrOverReco_vs_Pt_F->Fill(ijetPt, ratio);
        for (int i = 0; i < n_bins_pt; ++i) {
          if ((ijetPt > ptBins_[i]) && (ijetPt < ptBins_[i + 1]))
            hVector_JetPtCorrOverReco_F_ptBins[i]->Fill(ratio);
        }
      }
    }
  }

  if (!mEvent.isRealData()) {
    
    //----------------------------------------------------------------------------
    // Fill Gen Jets histograms
    //----------------------------------------------------------------------------

    edm::Handle<GenJetCollection> genJets;
    mEvent.getByToken(genJetsToken_, genJets);

    if (!genJets.isValid())
      return;

    if (!(mInputGenCollection.label().empty())) {
      for (GenJetCollection::const_iterator gjet = genJets->begin(); gjet != genJets->end(); gjet++) {
        // MiniAOD has intrinsic thresholds, introduce threshold for RECO too
        if (gjet->pt() < mMatchGenPtThreshold)
          continue;
        if (std::abs(gjet->eta()) > 6.)
          continue;  // Out of the detector

        if (mGenEta)
          mGenEta->Fill(gjet->eta());
        if (mGenPhi)
          mGenPhi->Fill(gjet->phi());
        if (mGenPt)
          mGenPt->Fill(gjet->pt());
        if (gjet == genJets->begin()) {
          if (mGenEtaFirst)
            mGenEtaFirst->Fill(gjet->eta());
          if (mGenPhiFirst)
            mGenPhiFirst->Fill(gjet->phi());
        }

        if (recoJets.empty())
          continue;

        //----------------------------------------------------------------------------
        // Match gen jets to reco jets
        //----------------------------------------------------------------------------

        int iMatchReco = -1;
        double deltaRBestReco = 999;
        for (unsigned ijet = 0; ijet < recoJets.size(); ++ijet) {
          if (recoJets[ijet].pt() > 10) {
            double dR = deltaR(gjet->eta(), gjet->phi(), recoJets[ijet].eta(), recoJets[ijet].phi());
            if (dR < deltaRBestReco) {
              iMatchReco = ijet;
              deltaRBestReco = dR;
            }
          }
        }
        
        if ((iMatchReco >= 0) && (deltaRBestReco < mRThreshold)) {

          //----------------------------------------------------------------------------
          // Fill gen jets to reco jets histograms
          //----------------------------------------------------------------------------

          double jec_factor = 1.0;
          if (isMiniAODJet) 
            jec_factor = (*patJets)[iMatchReco].jecFactor("Uncorrected");
          double response = (recoJets[iMatchReco].pt() * jec_factor) / gjet->pt();

          p_JetPtRecoOverGen_vs_GenEta->Fill(gjet->eta(), response);
          h2d_JetPtRecoOverGen_vs_GenEta->Fill(gjet->eta(), response);
          if (std::abs(gjet->eta()) < 1.5) {
            h_JetPtRecoOverGen_B->Fill(response);
            p_JetPtRecoOverGen_vs_GenPt_B->Fill(gjet->pt(), response);
            h2d_JetPtRecoOverGen_vs_GenPt_B->Fill(gjet->pt(), response);
            p_JetPtRecoOverGen_vs_GenPhi_B->Fill(gjet->phi(), response);
            h2d_JetPtRecoOverGen_vs_GenPhi_B->Fill(gjet->phi(), response);
            for (int i = 0; i < n_bins_pt; ++i) {
              if ((gjet->pt() > ptBins_[i]) && (gjet->pt() < ptBins_[i + 1]))
                hVector_JetPtRecoOverGen_B_ptBins[i]->Fill(response);
            }
          } else if (std::abs(gjet->eta()) < 3.0) {
            h_JetPtRecoOverGen_E->Fill(response);
            p_JetPtRecoOverGen_vs_GenPt_E->Fill(gjet->pt(), response);
            h2d_JetPtRecoOverGen_vs_GenPt_E->Fill(gjet->pt(), response);
            p_JetPtRecoOverGen_vs_GenPhi_E->Fill(gjet->phi(), response);
            h2d_JetPtRecoOverGen_vs_GenPhi_E->Fill(gjet->phi(), response);
            for (int i = 0; i < n_bins_pt; ++i) {
              if ((gjet->pt() > ptBins_[i]) && (gjet->pt() < ptBins_[i + 1]))
                hVector_JetPtRecoOverGen_E_ptBins[i]->Fill(response);
            }
          } else if (std::abs(gjet->eta()) < 6.0) {
            h_JetPtRecoOverGen_F->Fill(response);
            p_JetPtRecoOverGen_vs_GenPt_F->Fill(gjet->pt(), response);
            h2d_JetPtRecoOverGen_vs_GenPt_F->Fill(gjet->pt(), response);
            p_JetPtRecoOverGen_vs_GenPhi_F->Fill(gjet->phi(), response);
            h2d_JetPtRecoOverGen_vs_GenPhi_F->Fill(gjet->phi(), response);
            for (int i = 0; i < n_bins_pt; ++i) {
              if ((gjet->pt() > ptBins_[i]) && (gjet->pt() < ptBins_[i + 1]))
                hVector_JetPtRecoOverGen_F_ptBins[i]->Fill(response);
            }
          }

          //----------------------------------------------------------------------------
          // Fill gen jets to corrected jets histograms
          //----------------------------------------------------------------------------

          Jet MatchedCorrJet = recoJets[iMatchReco];
          if (pass_correction_flag && !isMiniAODJet) {
            if (isCaloJet)
              scale = jetCorr->correction((*caloJets)[iMatchReco]);
            if (isPFJet)
              scale = jetCorr->correction((*pfJets)[iMatchReco]);
            MatchedCorrJet.scaleEnergy(scale);
          }

          double responseCorr = (MatchedCorrJet.pt() * jec_factor) / gjet->pt();
          p_JetPtCorrOverGen_vs_GenEta->Fill(gjet->eta(), responseCorr);
          h2d_JetPtCorrOverGen_vs_GenEta->Fill(gjet->eta(), responseCorr);
          if (std::abs(gjet->eta()) < 1.5) {
            h_JetPtCorrOverGen_B->Fill(responseCorr);
            p_JetPtCorrOverGen_vs_GenPt_B->Fill(gjet->pt(), responseCorr);
            h2d_JetPtCorrOverGen_vs_GenPt_B->Fill(gjet->pt(), responseCorr);
            p_JetPtCorrOverGen_vs_GenPhi_B->Fill(gjet->phi(), responseCorr);
            h2d_JetPtCorrOverGen_vs_GenPhi_B->Fill(gjet->phi(), responseCorr);
            for (int i = 0; i < n_bins_pt; ++i) {
              if ((gjet->pt() > ptBins_[i]) && (gjet->pt() < ptBins_[i + 1]))
                hVector_JetPtCorrOverGen_B_ptBins[i]->Fill(responseCorr);
            }
          } else if (std::abs(gjet->eta()) < 3.0) {
            h_JetPtCorrOverGen_E->Fill(responseCorr);
            p_JetPtCorrOverGen_vs_GenPt_E->Fill(gjet->pt(), responseCorr);
            h2d_JetPtCorrOverGen_vs_GenPt_E->Fill(gjet->pt(), responseCorr);
            p_JetPtCorrOverGen_vs_GenPhi_E->Fill(gjet->phi(), responseCorr);
            h2d_JetPtCorrOverGen_vs_GenPhi_E->Fill(gjet->phi(), responseCorr);
            for (int i = 0; i < n_bins_pt; ++i) {
              if ((gjet->pt() > ptBins_[i]) && (gjet->pt() < ptBins_[i + 1]))
                hVector_JetPtCorrOverGen_E_ptBins[i]->Fill(responseCorr);
            }
          } else if (std::abs(gjet->eta()) < 6.0) {
            h_JetPtCorrOverGen_F->Fill(responseCorr);
            p_JetPtCorrOverGen_vs_GenPt_F->Fill(gjet->pt(), responseCorr);
            h2d_JetPtCorrOverGen_vs_GenPt_F->Fill(gjet->pt(), responseCorr);
            p_JetPtCorrOverGen_vs_GenPhi_F->Fill(gjet->phi(), responseCorr);
            h2d_JetPtCorrOverGen_vs_GenPhi_F->Fill(gjet->phi(), responseCorr);
            for (int i = 0; i < n_bins_pt; ++i) {
              if ((gjet->pt() > ptBins_[i]) && (gjet->pt() < ptBins_[i + 1]))
                hVector_JetPtCorrOverGen_F_ptBins[i]->Fill(responseCorr);
            }
          }

          mDeltaEta->Fill(gjet->eta() - MatchedCorrJet.eta());
          mDeltaPhi->Fill(gjet->phi() - MatchedCorrJet.phi());
          mDeltaPt->Fill((gjet->pt() - MatchedCorrJet.pt() * jec_factor) / gjet->pt());

        }
      }
    }
  }
}
