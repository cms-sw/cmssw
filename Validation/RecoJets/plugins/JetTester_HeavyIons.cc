// Producer for validation histograms for Calo and PF background subtracted
// objects Modified by Raghav Kunnawalkam Elayavalli, Aug 18th 2014
//                                          , Oct 22nd 2014 to run in 73X

#include "JetTester_HeavyIons.h"

using namespace edm;
using namespace reco;
using namespace std;

JetTester_HeavyIons::JetTester_HeavyIons(const edm::ParameterSet &iConfig)
    : mInputCollection(iConfig.getParameter<edm::InputTag>("src")),
      mInputGenCollection(iConfig.getParameter<edm::InputTag>("srcGen")),
      mInputPFCandCollection(iConfig.getParameter<edm::InputTag>("PFcands")),
      // mInputCandCollection           (iConfig.getParameter<edm::InputTag>
      // ("Cands")), rhoTag (iConfig.getParameter<edm::InputTag> ("srcRho")),
      mOutputFile(iConfig.getUntrackedParameter<std::string>("OutputFile", "")),
      JetType(iConfig.getUntrackedParameter<std::string>("JetType")),
      UEAlgo(iConfig.getUntrackedParameter<std::string>("UEAlgo")),
      Background(iConfig.getParameter<edm::InputTag>("Background")),
      mRecoJetPtThreshold(iConfig.getParameter<double>("recoJetPtThreshold")),
      mMatchGenPtThreshold(iConfig.getParameter<double>("matchGenPtThreshold")),
      mGenEnergyFractionThreshold(iConfig.getParameter<double>("genEnergyFractionThreshold")),
      mReverseEnergyFractionThreshold(iConfig.getParameter<double>("reverseEnergyFractionThreshold")),
      mRThreshold(iConfig.getParameter<double>("RThreshold")),
      JetCorrectionService(iConfig.getParameter<std::string>("JetCorrections")) {
  std::string inputCollectionLabel(mInputCollection.label());

  isCaloJet = (std::string("calo") == JetType);
  isJPTJet = (std::string("jpt") == JetType);
  isPFJet = (std::string("pf") == JetType);

  // consumes
  pvToken_ = consumes<std::vector<reco::Vertex>>(edm::InputTag("offlinePrimaryVertices"));
  caloTowersToken_ = consumes<CaloTowerCollection>(edm::InputTag("towerMaker"));
  if (isCaloJet)
    caloJetsToken_ = consumes<reco::CaloJetCollection>(mInputCollection);
  if (isJPTJet)
    jptJetsToken_ = consumes<reco::JPTJetCollection>(mInputCollection);
  if (isPFJet) {
    if (std::string("Pu") == UEAlgo)
      basicJetsToken_ = consumes<reco::BasicJetCollection>(mInputCollection);
  }

  genJetsToken_ = consumes<reco::GenJetCollection>(edm::InputTag(mInputGenCollection));
  evtToken_ = consumes<GenEventInfoProduct>(edm::InputTag("generator"));
  pfCandToken_ = consumes<reco::PFCandidateCollection>(mInputPFCandCollection);
  pfCandViewToken_ = consumes<reco::CandidateView>(mInputPFCandCollection);
  caloCandViewToken_ = consumes<reco::CandidateView>(edm::InputTag("towerMaker"));
  backgrounds_ = consumes<edm::ValueMap<reco::VoronoiBackground>>(Background);
  backgrounds_value_ = consumes<std::vector<float>>(Background);
  centralityTag_ = iConfig.getParameter<InputTag>("centralitycollection");
  centralityToken = consumes<reco::Centrality>(centralityTag_);

  centralityBinTag_ = (iConfig.getParameter<edm::InputTag>("centralitybincollection"));
  centralityBinToken = consumes<int>(centralityBinTag_);
  hiVertexToken_ = consumes<std::vector<reco::Vertex>>(edm::InputTag("hiSelectedVertex"));

  // need to initialize the PF cand histograms : which are also event variables
  if (isPFJet) {
    mNPFpart = nullptr;
    mPFPt = nullptr;
    mPFEta = nullptr;
    mPFPhi = nullptr;
    mPFArea = nullptr;
    mSumPFPt = nullptr;
    mSumSquaredPFPt = nullptr;
    mSumPFPt_HF = nullptr;

    mSumPFPt_n5p191_n2p650 = nullptr;
    mSumPFPt_n2p650_n2p043 = nullptr;
    mSumPFPt_n2p043_n1p740 = nullptr;
    mSumPFPt_n1p740_n1p479 = nullptr;
    mSumPFPt_n1p479_n1p131 = nullptr;
    mSumPFPt_n1p131_n0p783 = nullptr;
    mSumPFPt_n0p783_n0p522 = nullptr;
    mSumPFPt_n0p522_0p522 = nullptr;
    mSumPFPt_0p522_0p783 = nullptr;
    mSumPFPt_0p783_1p131 = nullptr;
    mSumPFPt_1p131_1p479 = nullptr;
    mSumPFPt_1p479_1p740 = nullptr;
    mSumPFPt_1p740_2p043 = nullptr;
    mSumPFPt_2p043_2p650 = nullptr;
    mSumPFPt_2p650_5p191 = nullptr;

    mPFCandpT_vs_eta_Unknown = nullptr;        // pf id 0
    mPFCandpT_vs_eta_ChargedHadron = nullptr;  // pf id - 1
    mPFCandpT_vs_eta_electron = nullptr;       // pf id - 2
    mPFCandpT_vs_eta_muon = nullptr;           // pf id - 3
    mPFCandpT_vs_eta_photon = nullptr;         // pf id - 4
    mPFCandpT_vs_eta_NeutralHadron = nullptr;  // pf id - 5
    mPFCandpT_vs_eta_HadE_inHF = nullptr;      // pf id - 6
    mPFCandpT_vs_eta_EME_inHF = nullptr;       // pf id - 7

    mPFCandpT_Barrel_Unknown = nullptr;        // pf id 0
    mPFCandpT_Barrel_ChargedHadron = nullptr;  // pf id - 1
    mPFCandpT_Barrel_electron = nullptr;       // pf id - 2
    mPFCandpT_Barrel_muon = nullptr;           // pf id - 3
    mPFCandpT_Barrel_photon = nullptr;         // pf id - 4
    mPFCandpT_Barrel_NeutralHadron = nullptr;  // pf id - 5
    mPFCandpT_Barrel_HadE_inHF = nullptr;      // pf id - 6
    mPFCandpT_Barrel_EME_inHF = nullptr;       // pf id - 7

    mPFCandpT_Endcap_Unknown = nullptr;        // pf id 0
    mPFCandpT_Endcap_ChargedHadron = nullptr;  // pf id - 1
    mPFCandpT_Endcap_electron = nullptr;       // pf id - 2
    mPFCandpT_Endcap_muon = nullptr;           // pf id - 3
    mPFCandpT_Endcap_photon = nullptr;         // pf id - 4
    mPFCandpT_Endcap_NeutralHadron = nullptr;  // pf id - 5
    mPFCandpT_Endcap_HadE_inHF = nullptr;      // pf id - 6
    mPFCandpT_Endcap_EME_inHF = nullptr;       // pf id - 7

    mPFCandpT_Forward_Unknown = nullptr;        // pf id 0
    mPFCandpT_Forward_ChargedHadron = nullptr;  // pf id - 1
    mPFCandpT_Forward_electron = nullptr;       // pf id - 2
    mPFCandpT_Forward_muon = nullptr;           // pf id - 3
    mPFCandpT_Forward_photon = nullptr;         // pf id - 4
    mPFCandpT_Forward_NeutralHadron = nullptr;  // pf id - 5
    mPFCandpT_Forward_HadE_inHF = nullptr;      // pf id - 6
    mPFCandpT_Forward_EME_inHF = nullptr;       // pf id - 7
  }
  if (isCaloJet) {
    mNCalopart = nullptr;
    mCaloPt = nullptr;
    mCaloEta = nullptr;
    mCaloPhi = nullptr;
    mCaloArea = nullptr;

    mSumCaloPt = nullptr;
    mSumSquaredCaloPt = nullptr;
    mSumCaloPt_HF = nullptr;

    mSumCaloPt_n5p191_n2p650 = nullptr;
    mSumCaloPt_n2p650_n2p043 = nullptr;
    mSumCaloPt_n2p043_n1p740 = nullptr;
    mSumCaloPt_n1p740_n1p479 = nullptr;
    mSumCaloPt_n1p479_n1p131 = nullptr;
    mSumCaloPt_n1p131_n0p783 = nullptr;
    mSumCaloPt_n0p783_n0p522 = nullptr;
    mSumCaloPt_n0p522_0p522 = nullptr;
    mSumCaloPt_0p522_0p783 = nullptr;
    mSumCaloPt_0p783_1p131 = nullptr;
    mSumCaloPt_1p131_1p479 = nullptr;
    mSumCaloPt_1p479_1p740 = nullptr;
    mSumCaloPt_1p740_2p043 = nullptr;
    mSumCaloPt_2p043_2p650 = nullptr;
    mSumCaloPt_2p650_5p191 = nullptr;
  }
  mSumpt = nullptr;

  // Events variables
  mNvtx = nullptr;
  mHF = nullptr;

  // Jet parameters
  mEta = nullptr;
  mPhi = nullptr;
  mEnergy = nullptr;
  mP = nullptr;
  mPt = nullptr;
  mMass = nullptr;
  mConstituents = nullptr;
  mJetArea = nullptr;
  mjetpileup = nullptr;
  mNJets_40 = nullptr;
  mNJets = nullptr;

  mGenEta = nullptr;
  mGenPhi = nullptr;
  mGenPt = nullptr;
  mPtHat = nullptr;

  mPtRecoOverGen_B_20_30_Cent_0_10 = nullptr;
  mPtRecoOverGen_E_20_30_Cent_0_10 = nullptr;
  mPtRecoOverGen_F_20_30_Cent_0_10 = nullptr;
  mPtRecoOverGen_B_30_50_Cent_0_10 = nullptr;
  mPtRecoOverGen_E_30_50_Cent_0_10 = nullptr;
  mPtRecoOverGen_F_30_50_Cent_0_10 = nullptr;
  mPtRecoOverGen_B_50_80_Cent_0_10 = nullptr;
  mPtRecoOverGen_E_50_80_Cent_0_10 = nullptr;
  mPtRecoOverGen_F_50_80_Cent_0_10 = nullptr;
  mPtRecoOverGen_B_80_120_Cent_0_10 = nullptr;
  mPtRecoOverGen_E_80_120_Cent_0_10 = nullptr;
  mPtRecoOverGen_F_80_120_Cent_0_10 = nullptr;
  mPtRecoOverGen_B_120_180_Cent_0_10 = nullptr;
  mPtRecoOverGen_E_120_180_Cent_0_10 = nullptr;
  mPtRecoOverGen_F_120_180_Cent_0_10 = nullptr;
  mPtRecoOverGen_B_180_300_Cent_0_10 = nullptr;
  mPtRecoOverGen_E_180_300_Cent_0_10 = nullptr;
  mPtRecoOverGen_F_180_300_Cent_0_10 = nullptr;
  mPtRecoOverGen_B_300_Inf_Cent_0_10 = nullptr;
  mPtRecoOverGen_E_300_Inf_Cent_0_10 = nullptr;
  mPtRecoOverGen_F_300_Inf_Cent_0_10 = nullptr;

  mPtRecoOverGen_B_20_30_Cent_10_30 = nullptr;
  mPtRecoOverGen_E_20_30_Cent_10_30 = nullptr;
  mPtRecoOverGen_F_20_30_Cent_10_30 = nullptr;
  mPtRecoOverGen_B_30_50_Cent_10_30 = nullptr;
  mPtRecoOverGen_E_30_50_Cent_10_30 = nullptr;
  mPtRecoOverGen_F_30_50_Cent_10_30 = nullptr;
  mPtRecoOverGen_B_50_80_Cent_10_30 = nullptr;
  mPtRecoOverGen_E_50_80_Cent_10_30 = nullptr;
  mPtRecoOverGen_F_50_80_Cent_10_30 = nullptr;
  mPtRecoOverGen_B_80_120_Cent_10_30 = nullptr;
  mPtRecoOverGen_E_80_120_Cent_10_30 = nullptr;
  mPtRecoOverGen_F_80_120_Cent_10_30 = nullptr;
  mPtRecoOverGen_B_120_180_Cent_10_30 = nullptr;
  mPtRecoOverGen_E_120_180_Cent_10_30 = nullptr;
  mPtRecoOverGen_F_120_180_Cent_10_30 = nullptr;
  mPtRecoOverGen_B_180_300_Cent_10_30 = nullptr;
  mPtRecoOverGen_E_180_300_Cent_10_30 = nullptr;
  mPtRecoOverGen_F_180_300_Cent_10_30 = nullptr;
  mPtRecoOverGen_B_300_Inf_Cent_10_30 = nullptr;
  mPtRecoOverGen_E_300_Inf_Cent_10_30 = nullptr;
  mPtRecoOverGen_F_300_Inf_Cent_10_30 = nullptr;

  mPtRecoOverGen_B_20_30_Cent_30_50 = nullptr;
  mPtRecoOverGen_E_20_30_Cent_30_50 = nullptr;
  mPtRecoOverGen_F_20_30_Cent_30_50 = nullptr;
  mPtRecoOverGen_B_30_50_Cent_30_50 = nullptr;
  mPtRecoOverGen_E_30_50_Cent_30_50 = nullptr;
  mPtRecoOverGen_F_30_50_Cent_30_50 = nullptr;
  mPtRecoOverGen_B_50_80_Cent_30_50 = nullptr;
  mPtRecoOverGen_E_50_80_Cent_30_50 = nullptr;
  mPtRecoOverGen_F_50_80_Cent_30_50 = nullptr;
  mPtRecoOverGen_B_80_120_Cent_30_50 = nullptr;
  mPtRecoOverGen_E_80_120_Cent_30_50 = nullptr;
  mPtRecoOverGen_F_80_120_Cent_30_50 = nullptr;
  mPtRecoOverGen_B_120_180_Cent_30_50 = nullptr;
  mPtRecoOverGen_E_120_180_Cent_30_50 = nullptr;
  mPtRecoOverGen_F_120_180_Cent_30_50 = nullptr;
  mPtRecoOverGen_B_180_300_Cent_30_50 = nullptr;
  mPtRecoOverGen_E_180_300_Cent_30_50 = nullptr;
  mPtRecoOverGen_F_180_300_Cent_30_50 = nullptr;
  mPtRecoOverGen_B_300_Inf_Cent_30_50 = nullptr;
  mPtRecoOverGen_E_300_Inf_Cent_30_50 = nullptr;
  mPtRecoOverGen_F_300_Inf_Cent_30_50 = nullptr;

  mPtRecoOverGen_B_20_30_Cent_50_80 = nullptr;
  mPtRecoOverGen_E_20_30_Cent_50_80 = nullptr;
  mPtRecoOverGen_F_20_30_Cent_50_80 = nullptr;
  mPtRecoOverGen_B_30_50_Cent_50_80 = nullptr;
  mPtRecoOverGen_E_30_50_Cent_50_80 = nullptr;
  mPtRecoOverGen_F_30_50_Cent_50_80 = nullptr;
  mPtRecoOverGen_B_50_80_Cent_50_80 = nullptr;
  mPtRecoOverGen_E_50_80_Cent_50_80 = nullptr;
  mPtRecoOverGen_F_50_80_Cent_50_80 = nullptr;
  mPtRecoOverGen_B_80_120_Cent_50_80 = nullptr;
  mPtRecoOverGen_E_80_120_Cent_50_80 = nullptr;
  mPtRecoOverGen_F_80_120_Cent_50_80 = nullptr;
  mPtRecoOverGen_B_120_180_Cent_50_80 = nullptr;
  mPtRecoOverGen_E_120_180_Cent_50_80 = nullptr;
  mPtRecoOverGen_F_120_180_Cent_50_80 = nullptr;
  mPtRecoOverGen_B_180_300_Cent_50_80 = nullptr;
  mPtRecoOverGen_E_180_300_Cent_50_80 = nullptr;
  mPtRecoOverGen_F_180_300_Cent_50_80 = nullptr;
  mPtRecoOverGen_B_300_Inf_Cent_50_80 = nullptr;
  mPtRecoOverGen_E_300_Inf_Cent_50_80 = nullptr;
  mPtRecoOverGen_F_300_Inf_Cent_50_80 = nullptr;

  mPtRecoOverGen_GenPt_B_Cent_0_10 = nullptr;
  mPtRecoOverGen_GenPt_E_Cent_0_10 = nullptr;
  mPtRecoOverGen_GenPt_F_Cent_0_10 = nullptr;
  mPtRecoOverGen_GenPt_B_Cent_10_30 = nullptr;
  mPtRecoOverGen_GenPt_E_Cent_10_30 = nullptr;
  mPtRecoOverGen_GenPt_F_Cent_10_30 = nullptr;
  mPtRecoOverGen_GenPt_B_Cent_30_50 = nullptr;
  mPtRecoOverGen_GenPt_E_Cent_30_50 = nullptr;
  mPtRecoOverGen_GenPt_F_Cent_30_50 = nullptr;
  mPtRecoOverGen_GenPt_B_Cent_50_80 = nullptr;
  mPtRecoOverGen_GenPt_E_Cent_50_80 = nullptr;
  mPtRecoOverGen_GenPt_F_Cent_50_80 = nullptr;

  mPtRecoOverGen_GenEta_20_30_Cent_0_10 = nullptr;
  mPtRecoOverGen_GenEta_30_50_Cent_0_10 = nullptr;
  mPtRecoOverGen_GenEta_50_80_Cent_0_10 = nullptr;
  mPtRecoOverGen_GenEta_80_120_Cent_0_10 = nullptr;
  mPtRecoOverGen_GenEta_120_180_Cent_0_10 = nullptr;
  mPtRecoOverGen_GenEta_180_300_Cent_0_10 = nullptr;
  mPtRecoOverGen_GenEta_300_Inf_Cent_0_10 = nullptr;

  mPtRecoOverGen_GenEta_20_30_Cent_10_30 = nullptr;
  mPtRecoOverGen_GenEta_30_50_Cent_10_30 = nullptr;
  mPtRecoOverGen_GenEta_50_80_Cent_10_30 = nullptr;
  mPtRecoOverGen_GenEta_80_120_Cent_10_30 = nullptr;
  mPtRecoOverGen_GenEta_120_180_Cent_10_30 = nullptr;
  mPtRecoOverGen_GenEta_180_300_Cent_10_30 = nullptr;
  mPtRecoOverGen_GenEta_300_Inf_Cent_10_30 = nullptr;

  mPtRecoOverGen_GenEta_20_30_Cent_30_50 = nullptr;
  mPtRecoOverGen_GenEta_30_50_Cent_30_50 = nullptr;
  mPtRecoOverGen_GenEta_50_80_Cent_30_50 = nullptr;
  mPtRecoOverGen_GenEta_80_120_Cent_30_50 = nullptr;
  mPtRecoOverGen_GenEta_120_180_Cent_30_50 = nullptr;
  mPtRecoOverGen_GenEta_180_300_Cent_30_50 = nullptr;
  mPtRecoOverGen_GenEta_300_Inf_Cent_30_50 = nullptr;

  mPtRecoOverGen_GenEta_20_30_Cent_50_80 = nullptr;
  mPtRecoOverGen_GenEta_30_50_Cent_50_80 = nullptr;
  mPtRecoOverGen_GenEta_50_80_Cent_50_80 = nullptr;
  mPtRecoOverGen_GenEta_80_120_Cent_50_80 = nullptr;
  mPtRecoOverGen_GenEta_120_180_Cent_50_80 = nullptr;
  mPtRecoOverGen_GenEta_180_300_Cent_50_80 = nullptr;
  mPtRecoOverGen_GenEta_300_Inf_Cent_50_80 = nullptr;
}

void JetTester_HeavyIons::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const &iRun, edm::EventSetup const &) {
  ibooker.setCurrentFolder("JetMET/JetValidation/" + mInputCollection.label());

  double log10PtMin = 0.50;
  double log10PtMax = 3.75;
  int log10PtBins = 26;

  static const size_t ncms_hcal_edge_pseudorapidity = 82 + 1;
  static const double cms_hcal_edge_pseudorapidity[ncms_hcal_edge_pseudorapidity] = {
      -5.191, -4.889, -4.716, -4.538, -4.363, -4.191, -4.013, -3.839, -3.664, -3.489, -3.314, -3.139, -2.964, -2.853,
      -2.650, -2.500, -2.322, -2.172, -2.043, -1.930, -1.830, -1.740, -1.653, -1.566, -1.479, -1.392, -1.305, -1.218,
      -1.131, -1.044, -0.957, -0.879, -0.783, -0.696, -0.609, -0.522, -0.435, -0.348, -0.261, -0.174, -0.087, 0.000,
      0.087,  0.174,  0.261,  0.348,  0.435,  0.522,  0.609,  0.696,  0.783,  0.879,  0.957,  1.044,  1.131,  1.218,
      1.305,  1.392,  1.479,  1.566,  1.653,  1.740,  1.830,  1.930,  2.043,  2.172,  2.322,  2.500,  2.650,  2.853,
      2.964,  3.139,  3.314,  3.489,  3.664,  3.839,  4.013,  4.191,  4.363,  4.538,  4.716,  4.889,  5.191};

  double etaRange[91] = {-6.0, -5.8, -5.6, -5.4, -5.2, -5.0, -4.8, -4.6, -4.4, -4.2, -4.0, -3.8, -3.6, -3.4, -3.2, -3.0,
                         -2.9, -2.8, -2.7, -2.6, -2.5, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4,
                         -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,  0.1,  0.2,
                         0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1.0,  1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,
                         1.9,  2.0,  2.1,  2.2,  2.3,  2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  3.0,  3.2,  3.4,  3.6,  3.8,
                         4.0,  4.2,  4.4,  4.6,  4.8,  5.0,  5.2,  5.4,  5.6,  5.8,  6.0};

  double edge_pseudorapidity[etaBins_ + 1] = {-5.191,
                                              -2.650,
                                              -2.043,
                                              -1.740,
                                              -1.479,
                                              -1.131,
                                              -0.783,
                                              -0.522,
                                              0.522,
                                              0.783,
                                              1.131,
                                              1.479,
                                              1.740,
                                              2.043,
                                              2.650,
                                              5.191};

  TH2F *h2D_etabins_vs_pt2 = new TH2F(
      "h2D_etabins_vs_pt2", "etaBins (x axis), sum pt^{2} (y axis)", etaBins_, edge_pseudorapidity, 10000, 0, 10000);
  TH2F *h2D_etabins_vs_pt = new TH2F(
      "h2D_etabins_vs_pt", "etaBins (x axis), sum pt (y axis)", etaBins_, edge_pseudorapidity, 10000, -1000, 1000);
  TH2F *h2D_etabins_vs_phi = new TH2F("h2D_etabins_vs_phi",
                                      "candidate map, eta(x axis), phi (y axis), pt (z axis)",
                                      ncms_hcal_edge_pseudorapidity - 1,
                                      cms_hcal_edge_pseudorapidity,
                                      36,
                                      -TMath::Pi(),
                                      TMath::Pi());
  TH2F *h2D_pfcand_etabins_vs_pt =
      new TH2F("h2D_etabins_vs_pt", ";#eta;sum p_{T}", etaBins_, edge_pseudorapidity, 300, 0, 300);

  if (isPFJet) {
    mNPFpart = ibooker.book1D("NPFpart", "No of particle flow candidates", 1000, 0, 10000);
    mPFPt = ibooker.book1D("PFPt", "PF candidate p_{T}", 1000, -5000, 5000);
    mPFEta = ibooker.book1D("PFEta", "PF candidate #eta", 120, -6, 6);
    mPFPhi = ibooker.book1D("PFPhi", "PF candidate #phi", 70, -3.5, 3.5);
    mPFArea = ibooker.book1D("PFArea", "VS PF candidate area", 100, 0, 4);

    mSumPFPt = ibooker.book1D("SumPFPt", "Sum of initial PF p_{T}", 1000, -10000, 10000);

    mSumSquaredPFPt = ibooker.book1D("SumSquaredPFPt", "Sum of initial PF p_{T} squared", 10000, 0, 10000);

    mSumPFPt_HF = ibooker.book2D(
        "SumPFPt_HF", "HF energy (y axis) vs Sum initial PF p_{T} (x axis)", 1000, -1000, 1000, 1000, 0, 10000);

    mSumPFPt_n5p191_n2p650 =
        ibooker.book1D("mSumPFPt_n5p191_n2p650", "Sum PFPt  in the eta range -5.191 to -2.650", 1000, -5000, 5000);
    mSumPFPt_n2p650_n2p043 =
        ibooker.book1D("mSumPFPt_n2p650_n2p043", "Sum PFPt  in the eta range -2.650 to -2.043 ", 1000, -5000, 5000);
    mSumPFPt_n2p043_n1p740 =
        ibooker.book1D("mSumPFPt_n2p043_n1p740", "Sum PFPt  in the eta range -2.043 to -1.740", 1000, -1000, 1000);
    mSumPFPt_n1p740_n1p479 =
        ibooker.book1D("mSumPFPt_n1p740_n1p479", "Sum PFPt  in the eta range -1.740 to -1.479", 1000, -1000, 1000);
    mSumPFPt_n1p479_n1p131 =
        ibooker.book1D("mSumPFPt_n1p479_n1p131", "Sum PFPt  in the eta range -1.479 to -1.131", 1000, -1000, 1000);
    mSumPFPt_n1p131_n0p783 =
        ibooker.book1D("mSumPFPt_n1p131_n0p783", "Sum PFPt  in the eta range -1.131 to -0.783", 1000, -1000, 1000);
    mSumPFPt_n0p783_n0p522 =
        ibooker.book1D("mSumPFPt_n0p783_n0p522", "Sum PFPt  in the eta range -0.783 to -0.522", 1000, -1000, 1000);
    mSumPFPt_n0p522_0p522 =
        ibooker.book1D("mSumPFPt_n0p522_0p522", "Sum PFPt  in the eta range -0.522 to 0.522", 1000, -1000, 1000);
    mSumPFPt_0p522_0p783 =
        ibooker.book1D("mSumPFPt_0p522_0p783", "Sum PFPt  in the eta range 0.522 to 0.783", 1000, -1000, 1000);
    mSumPFPt_0p783_1p131 =
        ibooker.book1D("mSumPFPt_0p783_1p131", "Sum PFPt  in the eta range 0.783 to 1.131", 1000, -1000, 1000);
    mSumPFPt_1p131_1p479 =
        ibooker.book1D("mSumPFPt_1p131_1p479", "Sum PFPt  in the eta range 1.131 to 1.479", 1000, -1000, 1000);
    mSumPFPt_1p479_1p740 =
        ibooker.book1D("mSumPFPt_1p479_1p740", "Sum PFPt  in the eta range 1.479 to 1.740", 1000, -1000, 1000);
    mSumPFPt_1p740_2p043 =
        ibooker.book1D("mSumPFPt_1p740_2p043", "Sum PFPt  in the eta range 1.740 to 2.043", 1000, -1000, 1000);
    mSumPFPt_2p043_2p650 =
        ibooker.book1D("mSumPFPt_2p043_2p650", "Sum PFPt  in the eta range 2.043 to 2.650", 1000, -5000, 5000);
    mSumPFPt_2p650_5p191 =
        ibooker.book1D("mSumPFPt_2p650_5p191", "Sum PFPt  in the eta range 2.650 to 5.191", 1000, -5000, 5000);

    mPFCandpT_vs_eta_Unknown = ibooker.book2D("PF_cand_X_unknown", h2D_pfcand_etabins_vs_pt);         // pf id 0
    mPFCandpT_vs_eta_ChargedHadron = ibooker.book2D("PF_cand_chargedHad", h2D_pfcand_etabins_vs_pt);  // pf id - 1
    mPFCandpT_vs_eta_electron = ibooker.book2D("PF_cand_electron", h2D_pfcand_etabins_vs_pt);         // pf id - 2
    mPFCandpT_vs_eta_muon = ibooker.book2D("PF_cand_muon", h2D_pfcand_etabins_vs_pt);                 // pf id - 3
    mPFCandpT_vs_eta_photon = ibooker.book2D("PF_cand_photon", h2D_pfcand_etabins_vs_pt);             // pf id - 4
    mPFCandpT_vs_eta_NeutralHadron = ibooker.book2D("PF_cand_neutralHad", h2D_pfcand_etabins_vs_pt);  // pf id - 5
    mPFCandpT_vs_eta_HadE_inHF = ibooker.book2D("PF_cand_HadEner_inHF", h2D_pfcand_etabins_vs_pt);    // pf id - 6
    mPFCandpT_vs_eta_EME_inHF = ibooker.book2D("PF_cand_EMEner_inHF", h2D_pfcand_etabins_vs_pt);      // pf id - 7

    mPFCandpT_Barrel_Unknown = ibooker.book1D("mPFCandpT_Barrel_Unknown",
                                              Form(";PF candidate p_{T}, |#eta|<%2.2f; counts", BarrelEta),
                                              300,
                                              0,
                                              300);  // pf id  - 0
    mPFCandpT_Barrel_ChargedHadron = ibooker.book1D("mPFCandpT_Barrel_ChargedHadron",
                                                    Form(";PF candidate p_{T}, |#eta|<%2.2f; counts", BarrelEta),
                                                    300,
                                                    0,
                                                    300);  // pf id - 1
    mPFCandpT_Barrel_electron = ibooker.book1D("mPFCandpT_Barrel_electron",
                                               Form(";PF candidate p_{T}, |#eta|<%2.2f; counts", BarrelEta),
                                               300,
                                               0,
                                               300);  // pf id - 2
    mPFCandpT_Barrel_muon = ibooker.book1D("mPFCandpT_Barrel_muon",
                                           Form(";PF candidate p_{T}, |#eta|<%2.2f; counts", BarrelEta),
                                           300,
                                           0,
                                           300);  // pf id - 3
    mPFCandpT_Barrel_photon = ibooker.book1D("mPFCandpT_Barrel_photon",
                                             Form(";PF candidate p_{T}, |#eta|<%2.2f; counts", BarrelEta),
                                             300,
                                             0,
                                             300);  // pf id - 4
    mPFCandpT_Barrel_NeutralHadron = ibooker.book1D("mPFCandpT_Barrel_NeutralHadron",
                                                    Form(";PF candidate p_{T}, |#eta|<%2.2f; counts", BarrelEta),
                                                    300,
                                                    0,
                                                    300);  // pf id - 5
    mPFCandpT_Barrel_HadE_inHF = ibooker.book1D("mPFCandpT_Barrel_HadE_inHF",
                                                Form(";PF candidate p_{T}, |#eta|<%2.2f; counts", BarrelEta),
                                                300,
                                                0,
                                                300);  // pf id - 6
    mPFCandpT_Barrel_EME_inHF = ibooker.book1D("mPFCandpT_Barrel_EME_inHF",
                                               Form(";PF candidate p_{T}, |#eta|<%2.2f; counts", BarrelEta),
                                               300,
                                               0,
                                               300);  // pf id - 7

    mPFCandpT_Endcap_Unknown =
        ibooker.book1D("mPFCandpT_Endcap_Unknown",
                       Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", BarrelEta, EndcapEta),
                       300,
                       0,
                       300);  // pf id - 0
    mPFCandpT_Endcap_ChargedHadron =
        ibooker.book1D("mPFCandpT_Endcap_ChargedHadron",
                       Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", BarrelEta, EndcapEta),
                       300,
                       0,
                       300);  // pf id - 1
    mPFCandpT_Endcap_electron =
        ibooker.book1D("mPFCandpT_Endcap_electron",
                       Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", BarrelEta, EndcapEta),
                       300,
                       0,
                       300);  // pf id - 2
    mPFCandpT_Endcap_muon =
        ibooker.book1D("mPFCandpT_Endcap_muon",
                       Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", BarrelEta, EndcapEta),
                       300,
                       0,
                       300);  // pf id - 3
    mPFCandpT_Endcap_photon =
        ibooker.book1D("mPFCandpT_Endcap_photon",
                       Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", BarrelEta, EndcapEta),
                       300,
                       0,
                       300);  // pf id - 4
    mPFCandpT_Endcap_NeutralHadron =
        ibooker.book1D("mPFCandpT_Endcap_NeutralHadron",
                       Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", BarrelEta, EndcapEta),
                       300,
                       0,
                       300);  // pf id - 5
    mPFCandpT_Endcap_HadE_inHF =
        ibooker.book1D("mPFCandpT_Endcap_HadE_inHF",
                       Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", BarrelEta, EndcapEta),
                       300,
                       0,
                       300);  // pf id - 6
    mPFCandpT_Endcap_EME_inHF =
        ibooker.book1D("mPFCandpT_Endcap_EME_inHF",
                       Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", BarrelEta, EndcapEta),
                       300,
                       0,
                       300);  // pf id - 7

    mPFCandpT_Forward_Unknown =
        ibooker.book1D("mPFCandpT_Forward_Unknown",
                       Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", EndcapEta, ForwardEta),
                       300,
                       0,
                       300);  // pf id - 0
    mPFCandpT_Forward_ChargedHadron =
        ibooker.book1D("mPFCandpT_Forward_ChargedHadron",
                       Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", EndcapEta, ForwardEta),
                       300,
                       0,
                       300);  // pf id - 1
    mPFCandpT_Forward_electron =
        ibooker.book1D("mPFCandpT_Forward_electron",
                       Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", EndcapEta, ForwardEta),
                       300,
                       0,
                       300);  // pf id - 2
    mPFCandpT_Forward_muon =
        ibooker.book1D("mPFCandpT_Forward_muon",
                       Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", EndcapEta, ForwardEta),
                       300,
                       0,
                       300);  // pf id - 3
    mPFCandpT_Forward_photon =
        ibooker.book1D("mPFCandpT_Forward_photon",
                       Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", EndcapEta, ForwardEta),
                       300,
                       0,
                       300);  // pf id - 4
    mPFCandpT_Forward_NeutralHadron =
        ibooker.book1D("mPFCandpT_Forward_NeutralHadron",
                       Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", EndcapEta, ForwardEta),
                       300,
                       0,
                       300);  // pf id - 5
    mPFCandpT_Forward_HadE_inHF =
        ibooker.book1D("mPFCandpT_Forward_HadE_inHF",
                       Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", EndcapEta, ForwardEta),
                       300,
                       0,
                       300);  // pf id - 6
    mPFCandpT_Forward_EME_inHF =
        ibooker.book1D("mPFCandpT_Forward_EME_inHF",
                       Form(";PF candidate p_{T}, %2.2f<|#eta|<%2.2f; counts", EndcapEta, ForwardEta),
                       300,
                       0,
                       300);  // pf id - 7
  }

  if (isCaloJet) {
    mNCalopart = ibooker.book1D("NCalopart", "No of particle flow candidates", 1000, 0, 10000);
    mCaloPt = ibooker.book1D("CaloPt", "Calo candidate p_{T}", 1000, -5000, 5000);
    mCaloEta = ibooker.book1D("CaloEta", "Calo candidate #eta", 120, -6, 6);
    mCaloPhi = ibooker.book1D("CaloPhi", "Calo candidate #phi", 70, -3.5, 3.5);
    mCaloArea = ibooker.book1D("CaloArea", "VS Calo candidate area", 100, 0, 4);

    mSumCaloPt = ibooker.book1D("SumCaloPt", "Sum Calo p_{T}", 1000, -10000, 10000);

    mSumSquaredCaloPt = ibooker.book1D("SumSquaredCaloPt", "Sum of initial Calo tower p_{T} squared", 10000, 0, 10000);

    mSumCaloPt_HF =
        ibooker.book2D("SumCaloPt_HF", "HF Energy (y axis) vs Sum Calo tower p_{T}", 1000, -1000, 1000, 1000, 0, 10000);

    mSumCaloPt_n5p191_n2p650 = ibooker.book1D(
        "mSumCaloPt_n5p191_n2p650", "Sum Calo tower pT variable in the eta range -5.191 to -2.650", 1000, -5000, 5000);
    mSumCaloPt_n2p650_n2p043 = ibooker.book1D(
        "mSumCaloPt_n2p650_n2p043", "Sum Calo tower pT variable in the eta range -2.650 to -2.043", 1000, -5000, 5000);
    mSumCaloPt_n2p043_n1p740 = ibooker.book1D(
        "mSumCaloPt_n2p043_n1p740", "Sum Calo tower pT variable in the eta range -2.043 to -1.740", 1000, -1000, 1000);
    mSumCaloPt_n1p740_n1p479 = ibooker.book1D(
        "mSumCaloPt_n1p740_n1p479", "Sum Calo tower pT variable in the eta range -1.740 to -1.479", 1000, -1000, 1000);
    mSumCaloPt_n1p479_n1p131 = ibooker.book1D(
        "mSumCaloPt_n1p479_n1p131", "Sum Calo tower pT variable in the eta range -1.479 to -1.131", 1000, -1000, 1000);
    mSumCaloPt_n1p131_n0p783 = ibooker.book1D(
        "mSumCaloPt_n1p131_n0p783", "Sum Calo tower pT variable in the eta range -1.131 to -0.783", 1000, -1000, 1000);
    mSumCaloPt_n0p783_n0p522 = ibooker.book1D(
        "mSumCaloPt_n0p783_n0p522", "Sum Calo tower pT variable in the eta range -0.783 to -0.522", 1000, -1000, 1000);
    mSumCaloPt_n0p522_0p522 = ibooker.book1D(
        "mSumCaloPt_n0p522_0p522", "Sum Calo tower pT variable in the eta range -0.522 to 0.522", 1000, -1000, 1000);
    mSumCaloPt_0p522_0p783 = ibooker.book1D(
        "mSumCaloPt_0p522_0p783", "Sum Calo tower pT variable in the eta range 0.522 to 0.783", 1000, -1000, 1000);
    mSumCaloPt_0p783_1p131 = ibooker.book1D(
        "mSumCaloPt_0p783_1p131", "Sum Calo tower pT variable in the eta range 0.783 to 1.131", 1000, -1000, 1000);
    mSumCaloPt_1p131_1p479 = ibooker.book1D(
        "mSumCaloPt_1p131_1p479", "Sum Calo tower pT variable in the eta range 1.131 to 1.479", 1000, -1000, 1000);
    mSumCaloPt_1p479_1p740 = ibooker.book1D(
        "mSumCaloPt_1p479_1p740", "Sum Calo tower pT variable in the eta range 1.479 to 1.740", 1000, -1000, 1000);
    mSumCaloPt_1p740_2p043 = ibooker.book1D(
        "mSumCaloPt_1p740_2p043", "Sum Calo tower pT variable in the eta range 1.740 to 2.043", 1000, -1000, 1000);
    mSumCaloPt_2p043_2p650 = ibooker.book1D(
        "mSumCaloPt_2p043_2p650", "Sum Calo tower pT variable in the eta range 2.043 to 2.650", 1000, -5000, 5000);
    mSumCaloPt_2p650_5p191 = ibooker.book1D(
        "mSumCaloPt_2p650_5p191", "Sum Calo tower pT variable in the eta range 2.650 to 5.191", 1000, -5000, 5000);
  }

  // particle flow variables histograms
  mSumpt = ibooker.book1D("SumpT", "Sum p_{T} of all the PF candidates per event", 1000, 0, 10000);

  // Event variables
  mNvtx = ibooker.book1D("Nvtx", "number of vertices", 60, 0, 60);
  mHF = ibooker.book1D("HF", "HF energy distribution", 1000, 0, 10000);

  // Jet parameters
  mEta = ibooker.book1D("Eta", "Eta", 120, -6, 6);
  mPhi = ibooker.book1D("Phi", "Phi", 70, -3.5, 3.5);
  mPt = ibooker.book1D("Pt", "Pt", 100, 0, 1000);
  mP = ibooker.book1D("P", "P", 100, 0, 1000);
  mEnergy = ibooker.book1D("Energy", "Energy", 100, 0, 1000);
  mMass = ibooker.book1D("Mass", "Mass", 100, 0, 200);
  mConstituents = ibooker.book1D("Constituents", "Constituents", 100, 0, 100);
  mJetArea = ibooker.book1D("JetArea", "JetArea", 100, 0, 4);
  mjetpileup = ibooker.book1D("jetPileUp", "jetPileUp", 100, 0, 150);
  mNJets_40 = ibooker.book1D("NJets_pt_greater_40", "NJets pT > 40 GeV", 50, 0, 100);
  mNJets = ibooker.book1D("NJets", "NJets", 50, 0, 100);

  mGenEta = ibooker.book1D("Gen Eta", ";gen jet #eta;counts", 120, -6, 6);
  mGenPhi = ibooker.book1D("Gen Phi", ";gen jet #phi;counts", 70, -3.5, 3.5);
  mGenPt = ibooker.book1D("Gen pT", "gen jet p_{T}", 250, 0, 1000);
  mPtHat = ibooker.book1D("pThat", "#hat{p_{T}}", 250, 0, 1000);

  mPtRecoOverGen_B_20_30_Cent_0_10 =
      ibooker.book1D("PtRecoOverGen_B_20_30_Cent_0_10", "20<genpt<30; recopt/genpt (0-10%) (Barrel);counts", 90, 0, 2);
  mPtRecoOverGen_E_20_30_Cent_0_10 =
      ibooker.book1D("PtRecoOverGen_E_20_30_Cent_0_10", "20<genpt<30; recopt/genpt (0-10%) (EndCap);counts", 90, 0, 2);
  mPtRecoOverGen_F_20_30_Cent_0_10 =
      ibooker.book1D("PtRecoOverGen_F_20_30_Cent_0_10", "20<genpt<30; recopt/genpt (0-10%) (Forward);counts", 90, 0, 2);
  mPtRecoOverGen_B_30_50_Cent_0_10 =
      ibooker.book1D("PtRecoOverGen_B_30_50_Cent_0_10", "30<genpt<50; recopt/genpt (0-10%) (Barrel);counts", 90, 0, 2);
  mPtRecoOverGen_E_30_50_Cent_0_10 =
      ibooker.book1D("PtRecoOverGen_E_30_50_Cent_0_10", "30<genpt<50; recopt/genpt (0-10%) (EndCap);counts", 90, 0, 2);
  mPtRecoOverGen_F_30_50_Cent_0_10 =
      ibooker.book1D("PtRecoOverGen_F_30_50_Cent_0_10", "30<genpt<50; recopt/genpt (0-10%) (Forward);counts", 90, 0, 2);
  mPtRecoOverGen_B_50_80_Cent_0_10 =
      ibooker.book1D("PtRecoOverGen_B_50_80_Cent_0_10", "50<genpt<80; recopt/genpt (0-10%) (Barrel);counts", 90, 0, 2);
  mPtRecoOverGen_E_50_80_Cent_0_10 =
      ibooker.book1D("PtRecoOverGen_E_50_80_Cent_0_10", "50<genpt<80; recopt/genpt (0-10%) (EndCap);counts", 90, 0, 2);
  mPtRecoOverGen_F_50_80_Cent_0_10 =
      ibooker.book1D("PtRecoOverGen_F_50_80_Cent_0_10", "50<genpt<80; recopt/genpt (0-10%) (Forward);counts", 90, 0, 2);
  mPtRecoOverGen_B_80_120_Cent_0_10 = ibooker.book1D(
      "PtRecoOverGen_B_80_120_Cent_0_10", "80<genpt<120; recopt/genpt (0-10%) (Barrel);counts", 90, 0, 2);
  mPtRecoOverGen_E_80_120_Cent_0_10 = ibooker.book1D(
      "PtRecoOverGen_E_80_120_Cent_0_10", "80<genpt<120; recopt/genpt (0-10%) (EndCap);counts", 90, 0, 2);
  mPtRecoOverGen_F_80_120_Cent_0_10 = ibooker.book1D(
      "PtRecoOverGen_F_80_120_Cent_0_10", "80<genpt<120; recopt/genpt (0-10%) (Forward);counts", 90, 0, 2);
  mPtRecoOverGen_B_120_180_Cent_0_10 = ibooker.book1D(
      "PtRecoOverGen_B_120_180_Cent_0_10", "120<genpt<180; recopt/genpt (0-10%) (Barrel);counts", 90, 0, 2);
  mPtRecoOverGen_E_120_180_Cent_0_10 = ibooker.book1D(
      "PtRecoOverGen_E_120_180_Cent_0_10", "120<genpt<180; recopt/genpt (0-10%) (EndCap);counts", 90, 0, 2);
  mPtRecoOverGen_F_120_180_Cent_0_10 = ibooker.book1D(
      "PtRecoOverGen_F_120_180_Cent_0_10", "120<genpt<180; recopt/genpt (0-10%) (Forward);counts", 90, 0, 2);
  mPtRecoOverGen_B_180_300_Cent_0_10 = ibooker.book1D(
      "PtRecoOverGen_B_180_300_Cent_0_10", "180<genpt<300; recopt/genpt (0-10%) (Barrel);counts", 90, 0, 2);
  mPtRecoOverGen_E_180_300_Cent_0_10 = ibooker.book1D(
      "PtRecoOverGen_E_180_300_Cent_0_10", "180<genpt<300; recopt/genpt (0-10%) (EndCap);counts", 90, 0, 2);
  mPtRecoOverGen_F_180_300_Cent_0_10 = ibooker.book1D(
      "PtRecoOverGen_F_180_300_Cent_0_10", "180<genpt<300; recopt/genpt (0-10%) (Forward);counts", 90, 0, 2);
  mPtRecoOverGen_B_300_Inf_Cent_0_10 = ibooker.book1D(
      "PtRecoOverGen_B_300_Inf_Cent_0_10", "300<genpt<Inf; recopt/genpt (0-10%) (Barrel);counts", 90, 0, 2);
  mPtRecoOverGen_E_300_Inf_Cent_0_10 = ibooker.book1D(
      "PtRecoOverGen_E_300_Inf_Cent_0_10", "300<genpt<Inf; recopt/genpt (0-10%) (EndCap);counts", 90, 0, 2);
  mPtRecoOverGen_F_300_Inf_Cent_0_10 = ibooker.book1D(
      "PtRecoOverGen_F_300_Inf_Cent_0_10", "300<genpt<Inf; recopt/genpt (0-10%) (Forward);counts", 90, 0, 2);

  mPtRecoOverGen_B_20_30_Cent_10_30 = ibooker.book1D(
      "PtRecoOverGen_B_20_30_Cent_10_30", "20<genpt<30; recopt/genpt (10-30%) (Barrel);counts", 90, 0, 2);
  mPtRecoOverGen_E_20_30_Cent_10_30 = ibooker.book1D(
      "PtRecoOverGen_E_20_30_Cent_10_30", "20<genpt<30; recopt/genpt (10-30%) (EndCap);counts", 90, 0, 2);
  mPtRecoOverGen_F_20_30_Cent_10_30 = ibooker.book1D(
      "PtRecoOverGen_F_20_30_Cent_10_30", "20<genpt<30; recopt/genpt (10-30%) (Forward);counts", 90, 0, 2);
  mPtRecoOverGen_B_30_50_Cent_10_30 = ibooker.book1D(
      "PtRecoOverGen_B_30_50_Cent_10_30", "30<genpt<50; recopt/genpt (10-30%) (Barrel);counts", 90, 0, 2);
  mPtRecoOverGen_E_30_50_Cent_10_30 = ibooker.book1D(
      "PtRecoOverGen_E_30_50_Cent_10_30", "30<genpt<50; recopt/genpt (10-30%) (EndCap);counts", 90, 0, 2);
  mPtRecoOverGen_F_30_50_Cent_10_30 = ibooker.book1D(
      "PtRecoOverGen_F_30_50_Cent_10_30", "30<genpt<50; recopt/genpt (10-30%) (Forward);counts", 90, 0, 2);
  mPtRecoOverGen_B_50_80_Cent_10_30 = ibooker.book1D(
      "PtRecoOverGen_B_50_80_Cent_10_30", "50<genpt<80; recopt/genpt (10-30%) (Barrel);counts", 90, 0, 2);
  mPtRecoOverGen_E_50_80_Cent_10_30 = ibooker.book1D(
      "PtRecoOverGen_E_50_80_Cent_10_30", "50<genpt<80; recopt/genpt (10-30%) (EndCap);counts", 90, 0, 2);
  mPtRecoOverGen_F_50_80_Cent_10_30 = ibooker.book1D(
      "PtRecoOverGen_F_50_80_Cent_10_30", "50<genpt<80; recopt/genpt (10-30%) (Forward);counts", 90, 0, 2);
  mPtRecoOverGen_B_80_120_Cent_10_30 = ibooker.book1D(
      "PtRecoOverGen_B_80_120_Cent_10_30", "80<genpt<120; recopt/genpt (10-30%) (Barrel);counts", 90, 0, 2);
  mPtRecoOverGen_E_80_120_Cent_10_30 = ibooker.book1D(
      "PtRecoOverGen_E_80_120_Cent_10_30", "80<genpt<120; recopt/genpt (10-30%) (EndCap);counts", 90, 0, 2);
  mPtRecoOverGen_F_80_120_Cent_10_30 = ibooker.book1D(
      "PtRecoOverGen_F_80_120_Cent_10_30", "80<genpt<120; recopt/genpt (10-30%) (Forward);counts", 90, 0, 2);
  mPtRecoOverGen_B_120_180_Cent_10_30 = ibooker.book1D(
      "PtRecoOverGen_B_120_180_Cent_10_30", "120<genpt<180; recopt/genpt (10-30%) (Barrel);counts", 90, 0, 2);
  mPtRecoOverGen_E_120_180_Cent_10_30 = ibooker.book1D(
      "PtRecoOverGen_E_120_180_Cent_10_30", "120<genpt<180; recopt/genpt (10-30%) (EndCap);counts", 90, 0, 2);
  mPtRecoOverGen_F_120_180_Cent_10_30 = ibooker.book1D(
      "PtRecoOverGen_F_120_180_Cent_10_30", "120<genpt<180; recopt/genpt (10-30%) (Forward);counts", 90, 0, 2);
  mPtRecoOverGen_B_180_300_Cent_10_30 = ibooker.book1D(
      "PtRecoOverGen_B_180_300_Cent_10_30", "180<genpt<300; recopt/genpt (10-30%) (Barrel);counts", 90, 0, 2);
  mPtRecoOverGen_E_180_300_Cent_10_30 = ibooker.book1D(
      "PtRecoOverGen_E_180_300_Cent_10_30", "180<genpt<300; recopt/genpt (10-30%) (EndCap);counts", 90, 0, 2);
  mPtRecoOverGen_F_180_300_Cent_10_30 = ibooker.book1D(
      "PtRecoOverGen_F_180_300_Cent_10_30", "180<genpt<300; recopt/genpt (10-30%) (Forward);counts", 90, 0, 2);
  mPtRecoOverGen_B_300_Inf_Cent_10_30 = ibooker.book1D(
      "PtRecoOverGen_B_300_Inf_Cent_10_30", "300<genpt<Inf; recopt/genpt (10-30%) (Barrel);counts", 90, 0, 2);
  mPtRecoOverGen_E_300_Inf_Cent_10_30 = ibooker.book1D(
      "PtRecoOverGen_E_300_Inf_Cent_10_30", "300<genpt<Inf; recopt/genpt (10-30%) (EndCap);counts", 90, 0, 2);
  mPtRecoOverGen_F_300_Inf_Cent_10_30 = ibooker.book1D(
      "PtRecoOverGen_F_300_Inf_Cent_10_30", "300<genpt<Inf; recopt/genpt (10-30%) (Forward);counts", 90, 0, 2);

  mPtRecoOverGen_B_20_30_Cent_30_50 = ibooker.book1D(
      "PtRecoOverGen_B_20_30_Cent_30_50", "20<genpt<30; recopt/genpt (30-50%) (Barrel);counts", 90, 0, 2);
  mPtRecoOverGen_E_20_30_Cent_30_50 = ibooker.book1D(
      "PtRecoOverGen_E_20_30_Cent_30_50", "20<genpt<30; recopt/genpt (30-50%) (EndCap);counts", 90, 0, 2);
  mPtRecoOverGen_F_20_30_Cent_30_50 = ibooker.book1D(
      "PtRecoOverGen_F_20_30_Cent_30_50", "20<genpt<30; recopt/genpt (30-50%) (Forward);counts", 90, 0, 2);
  mPtRecoOverGen_B_30_50_Cent_30_50 = ibooker.book1D(
      "PtRecoOverGen_B_30_50_Cent_30_50", "30<genpt<50; recopt/genpt (30-50%) (Barrel);counts", 90, 0, 2);
  mPtRecoOverGen_E_30_50_Cent_30_50 = ibooker.book1D(
      "PtRecoOverGen_E_30_50_Cent_30_50", "30<genpt<50; recopt/genpt (30-50%) (EndCap);counts", 90, 0, 2);
  mPtRecoOverGen_F_30_50_Cent_30_50 = ibooker.book1D(
      "PtRecoOverGen_F_30_50_Cent_30_50", "30<genpt<50; recopt/genpt (30-50%) (Forward);counts", 90, 0, 2);
  mPtRecoOverGen_B_50_80_Cent_30_50 = ibooker.book1D(
      "PtRecoOverGen_B_50_80_Cent_30_50", "50<genpt<80; recopt/genpt (30-50%) (Barrel);counts", 90, 0, 2);
  mPtRecoOverGen_E_50_80_Cent_30_50 = ibooker.book1D(
      "PtRecoOverGen_E_50_80_Cent_30_50", "50<genpt<80; recopt/genpt (30-50%) (EndCap);counts", 90, 0, 2);
  mPtRecoOverGen_F_50_80_Cent_30_50 = ibooker.book1D(
      "PtRecoOverGen_F_50_80_Cent_30_50", "50<genpt<80; recopt/genpt (30-50%) (Forward);counts", 90, 0, 2);
  mPtRecoOverGen_B_80_120_Cent_30_50 = ibooker.book1D(
      "PtRecoOverGen_B_80_120_Cent_30_50", "80<genpt<120; recopt/genpt (30-50%) (Barrel);counts", 90, 0, 2);
  mPtRecoOverGen_E_80_120_Cent_30_50 = ibooker.book1D(
      "PtRecoOverGen_E_80_120_Cent_30_50", "80<genpt<120; recopt/genpt (30-50%) (EndCap);counts", 90, 0, 2);
  mPtRecoOverGen_F_80_120_Cent_30_50 = ibooker.book1D(
      "PtRecoOverGen_F_80_120_Cent_30_50", "80<genpt<120; recopt/genpt (30-50%) (Forward);counts", 90, 0, 2);
  mPtRecoOverGen_B_120_180_Cent_30_50 = ibooker.book1D(
      "PtRecoOverGen_B_120_180_Cent_30_50", "120<genpt<180; recopt/genpt (30-50%) (Barrel);counts", 90, 0, 2);
  mPtRecoOverGen_E_120_180_Cent_30_50 = ibooker.book1D(
      "PtRecoOverGen_E_120_180_Cent_30_50", "120<genpt<180; recopt/genpt (30-50%) (EndCap);counts", 90, 0, 2);
  mPtRecoOverGen_F_120_180_Cent_30_50 = ibooker.book1D(
      "PtRecoOverGen_F_120_180_Cent_30_50", "120<genpt<180; recopt/genpt (30-50%) (Forward);counts", 90, 0, 2);
  mPtRecoOverGen_B_180_300_Cent_30_50 = ibooker.book1D(
      "PtRecoOverGen_B_180_300_Cent_30_50", "180<genpt<300; recopt/genpt (30-50%) (Barrel);counts", 90, 0, 2);
  mPtRecoOverGen_E_180_300_Cent_30_50 = ibooker.book1D(
      "PtRecoOverGen_E_180_300_Cent_30_50", "180<genpt<300; recopt/genpt (30-50%) (EndCap);counts", 90, 0, 2);
  mPtRecoOverGen_F_180_300_Cent_30_50 = ibooker.book1D(
      "PtRecoOverGen_F_180_300_Cent_30_50", "180<genpt<300; recopt/genpt (30-50%) (Forward);counts", 90, 0, 2);
  mPtRecoOverGen_B_300_Inf_Cent_30_50 = ibooker.book1D(
      "PtRecoOverGen_B_300_Inf_Cent_30_50", "300<genpt<Inf; recopt/genpt (30-50%) (Barrel);counts", 90, 0, 2);
  mPtRecoOverGen_E_300_Inf_Cent_30_50 = ibooker.book1D(
      "PtRecoOverGen_E_300_Inf_Cent_30_50", "300<genpt<Inf; recopt/genpt (30-50%) (EndCap);counts", 90, 0, 2);
  mPtRecoOverGen_F_300_Inf_Cent_30_50 = ibooker.book1D(
      "PtRecoOverGen_F_300_Inf_Cent_30_50", "300<genpt<Inf; recopt/genpt (30-50%) (Forward);counts", 90, 0, 2);

  mPtRecoOverGen_B_20_30_Cent_50_80 = ibooker.book1D(
      "PtRecoOverGen_B_20_30_Cent_50_80", "20<genpt<30; recopt/genpt (50-80%) (Barrel);counts", 90, 0, 2);
  mPtRecoOverGen_E_20_30_Cent_50_80 = ibooker.book1D(
      "PtRecoOverGen_E_20_30_Cent_50_80", "20<genpt<30; recopt/genpt (50-80%) (EndCap);counts", 90, 0, 2);
  mPtRecoOverGen_F_20_30_Cent_50_80 = ibooker.book1D(
      "PtRecoOverGen_F_20_30_Cent_50_80", "20<genpt<30; recopt/genpt (50-80%) (Forward);counts", 90, 0, 2);
  mPtRecoOverGen_B_30_50_Cent_50_80 = ibooker.book1D(
      "PtRecoOverGen_B_30_50_Cent_50_80", "30<genpt<50; recopt/genpt (50-80%) (Barrel);counts", 90, 0, 2);
  mPtRecoOverGen_E_30_50_Cent_50_80 = ibooker.book1D(
      "PtRecoOverGen_E_30_50_Cent_50_80", "30<genpt<50; recopt/genpt (50-80%) (EndCap);counts", 90, 0, 2);
  mPtRecoOverGen_F_30_50_Cent_50_80 = ibooker.book1D(
      "PtRecoOverGen_F_30_50_Cent_50_80", "30<genpt<50; recopt/genpt (50-80%) (Forward);counts", 90, 0, 2);
  mPtRecoOverGen_B_50_80_Cent_50_80 = ibooker.book1D(
      "PtRecoOverGen_B_50_80_Cent_50_80", "50<genpt<80; recopt/genpt (50-80%) (Barrel);counts", 90, 0, 2);
  mPtRecoOverGen_E_50_80_Cent_50_80 = ibooker.book1D(
      "PtRecoOverGen_E_50_80_Cent_50_80", "50<genpt<80; recopt/genpt (50-80%) (EndCap);counts", 90, 0, 2);
  mPtRecoOverGen_F_50_80_Cent_50_80 = ibooker.book1D(
      "PtRecoOverGen_F_50_80_Cent_50_80", "50<genpt<80; recopt/genpt (50-80%) (Forward);counts", 90, 0, 2);
  mPtRecoOverGen_B_80_120_Cent_50_80 = ibooker.book1D(
      "PtRecoOverGen_B_80_120_Cent_50_80", "80<genpt<120; recopt/genpt (50-80%) (Barrel);counts", 90, 0, 2);
  mPtRecoOverGen_E_80_120_Cent_50_80 = ibooker.book1D(
      "PtRecoOverGen_E_80_120_Cent_50_80", "80<genpt<120; recopt/genpt (50-80%) (EndCap);counts", 90, 0, 2);
  mPtRecoOverGen_F_80_120_Cent_50_80 = ibooker.book1D(
      "PtRecoOverGen_F_80_120_Cent_50_80", "80<genpt<120; recopt/genpt (50-80%) (Forward);counts", 90, 0, 2);
  mPtRecoOverGen_B_120_180_Cent_50_80 = ibooker.book1D(
      "PtRecoOverGen_B_120_180_Cent_50_80", "120<genpt<180; recopt/genpt (50-80%) (Barrel);counts", 90, 0, 2);
  mPtRecoOverGen_E_120_180_Cent_50_80 = ibooker.book1D(
      "PtRecoOverGen_E_120_180_Cent_50_80", "120<genpt<180; recopt/genpt (50-80%) (EndCap);counts", 90, 0, 2);
  mPtRecoOverGen_F_120_180_Cent_50_80 = ibooker.book1D(
      "PtRecoOverGen_F_120_180_Cent_50_80", "120<genpt<180; recopt/genpt (50-80%) (Forward);counts", 90, 0, 2);
  mPtRecoOverGen_B_180_300_Cent_50_80 = ibooker.book1D(
      "PtRecoOverGen_B_180_300_Cent_50_80", "180<genpt<300; recopt/genpt (50-80%) (Barrel);counts", 90, 0, 2);
  mPtRecoOverGen_E_180_300_Cent_50_80 = ibooker.book1D(
      "PtRecoOverGen_E_180_300_Cent_50_80", "180<genpt<300; recopt/genpt (50-80%) (EndCap);counts", 90, 0, 2);
  mPtRecoOverGen_F_180_300_Cent_50_80 = ibooker.book1D(
      "PtRecoOverGen_F_180_300_Cent_50_80", "180<genpt<300; recopt/genpt (50-80%) (Forward);counts", 90, 0, 2);
  mPtRecoOverGen_B_300_Inf_Cent_50_80 = ibooker.book1D(
      "PtRecoOverGen_B_300_Inf_Cent_50_80", "300<genpt<Inf; recopt/genpt (50-80%) (Barrel);counts", 90, 0, 2);
  mPtRecoOverGen_E_300_Inf_Cent_50_80 = ibooker.book1D(
      "PtRecoOverGen_E_300_Inf_Cent_50_80", "300<genpt<Inf; recopt/genpt (50-80%) (EndCap);counts", 90, 0, 2);
  mPtRecoOverGen_F_300_Inf_Cent_50_80 = ibooker.book1D(
      "PtRecoOverGen_F_300_Inf_Cent_50_80", "300<genpt<Inf; recopt/genpt (50-80%) (Forward);counts", 90, 0, 2);

  mPtRecoOverGen_GenPt_B_Cent_0_10 = ibooker.bookProfile("PtRecoOverGen_GenPt_B_Cent_0_10",
                                                         Form("|#eta|<%2.2f, (0-10cent);genpt;recopt/genpt", BarrelEta),
                                                         log10PtBins,
                                                         log10PtMin,
                                                         log10PtMax,
                                                         0,
                                                         2,
                                                         " ");
  mPtRecoOverGen_GenPt_E_Cent_0_10 =
      ibooker.bookProfile("PtRecoOverGen_GenPt_E_Cent_0_10",
                          Form("%2.2f<|#eta|<%2.2f, (0-10cent);genpt;recopt/genpt", BarrelEta, EndcapEta),
                          log10PtBins,
                          log10PtMin,
                          log10PtMax,
                          0,
                          2,
                          " ");
  mPtRecoOverGen_GenPt_F_Cent_0_10 =
      ibooker.bookProfile("PtRecoOverGen_GenPt_F_Cent_0_10",
                          Form("%2.2f<|#eta|<%2.2f, (0-10cent);genpt;recopt/genpt", EndcapEta, ForwardEta),
                          log10PtBins,
                          log10PtMin,
                          log10PtMax,
                          0,
                          2,
                          " ");
  mPtRecoOverGen_GenPt_B_Cent_10_30 =
      ibooker.bookProfile("PtRecoOverGen_GenPt_B_Cent_10_30",
                          Form("|#eta|<%2.2f, (10-30cent);genpt;recopt/genpt", BarrelEta),
                          log10PtBins,
                          log10PtMin,
                          log10PtMax,
                          0,
                          2,
                          " ");
  mPtRecoOverGen_GenPt_E_Cent_10_30 =
      ibooker.bookProfile("PtRecoOverGen_GenPt_E_Cent_10_30",
                          Form("%2.2f<|#eta|<%2.2f, (10-30cent);genpt;recopt/genpt", BarrelEta, EndcapEta),
                          log10PtBins,
                          log10PtMin,
                          log10PtMax,
                          0,
                          2,
                          " ");
  mPtRecoOverGen_GenPt_F_Cent_10_30 =
      ibooker.bookProfile("PtRecoOverGen_GenPt_F_Cent_10_30",
                          Form("%2.2f<|#eta|<%2.2f, (10-30cent);genpt;recopt/genpt", EndcapEta, ForwardEta),
                          log10PtBins,
                          log10PtMin,
                          log10PtMax,
                          0,
                          2,
                          " ");
  mPtRecoOverGen_GenPt_B_Cent_30_50 =
      ibooker.bookProfile("PtRecoOverGen_GenPt_B_Cent_30_50",
                          Form("|#eta|<%2.2f, (30-50cent);genpt;recopt/genpt", BarrelEta),
                          log10PtBins,
                          log10PtMin,
                          log10PtMax,
                          0,
                          2,
                          " ");
  mPtRecoOverGen_GenPt_E_Cent_30_50 =
      ibooker.bookProfile("PtRecoOverGen_GenPt_E_Cent_30_50",
                          Form("%2.2f<|#eta|<%2.2f, (30-50cent);genpt;recopt/genpt", BarrelEta, EndcapEta),
                          log10PtBins,
                          log10PtMin,
                          log10PtMax,
                          0,
                          2,
                          " ");
  mPtRecoOverGen_GenPt_F_Cent_30_50 =
      ibooker.bookProfile("PtRecoOverGen_GenPt_F_Cent_30_50",
                          Form("%2.2f<|#eta|<%2.2f, (30-50cent);genpt;recopt/genpt", EndcapEta, ForwardEta),
                          log10PtBins,
                          log10PtMin,
                          log10PtMax,
                          0,
                          2,
                          " ");
  mPtRecoOverGen_GenPt_B_Cent_50_80 =
      ibooker.bookProfile("PtRecoOverGen_GenPt_B_Cent_50_80",
                          Form("|#eta|<%2.2f, (50-80cent);genpt;recopt/genpt", BarrelEta),
                          log10PtBins,
                          log10PtMin,
                          log10PtMax,
                          0,
                          2,
                          " ");
  mPtRecoOverGen_GenPt_E_Cent_50_80 =
      ibooker.bookProfile("PtRecoOverGen_GenPt_E_Cent_50_80",
                          Form("%2.2f<|#eta|<%2.2f, (50-80cent);genpt;recopt/genpt", BarrelEta, EndcapEta),
                          log10PtBins,
                          log10PtMin,
                          log10PtMax,
                          0,
                          2,
                          " ");
  mPtRecoOverGen_GenPt_F_Cent_50_80 =
      ibooker.bookProfile("PtRecoOverGen_GenPt_F_Cent_50_80",
                          Form("%2.2f<|#eta|<%2.2f, (50-80cent);genpt;recopt/genpt", EndcapEta, ForwardEta),
                          log10PtBins,
                          log10PtMin,
                          log10PtMax,
                          0,
                          2,
                          " ");

  mPtRecoOverGen_GenEta_20_30_Cent_0_10 = ibooker.bookProfile(
      "PtRecoOverGen_GenEta_20_30_Cent_0_10", "20<genpt<30 (0-10%);geneta;recopt/genpt", 90, etaRange, 0, 2, " ");
  mPtRecoOverGen_GenEta_30_50_Cent_0_10 = ibooker.bookProfile(
      "PtRecoOverGen_GenEta_30_50_Cent_0_10", "30<genpt<50 (0-10%);geneta;recopt/genpt", 90, etaRange, 0, 2, " ");
  mPtRecoOverGen_GenEta_50_80_Cent_0_10 = ibooker.bookProfile(
      "PtRecoOverGen_GenEta_50_80_Cent_0_10", "50<genpt<80 (0-10%);geneta;recopt/genpt", 90, etaRange, 0, 2, " ");
  mPtRecoOverGen_GenEta_80_120_Cent_0_10 = ibooker.bookProfile(
      "PtRecoOverGen_GenEta_80_120_Cent_0_10", "80<genpt<120 (0-10%);geneta;recopt/genpt", 90, etaRange, 0, 2, " ");
  mPtRecoOverGen_GenEta_120_180_Cent_0_10 = ibooker.bookProfile(
      "PtRecoOverGen_GenEta_120_180_Cent_0_10", "120<genpt<180 (0-10%);geneta;recopt/genpt", 90, etaRange, 0, 2, " ");
  mPtRecoOverGen_GenEta_180_300_Cent_0_10 = ibooker.bookProfile(
      "PtRecoOverGen_GenEta_180_300_Cent_0_10", "180<genpt<300 (0-10%);geneta;recopt/genpt", 90, etaRange, 0, 2, " ");
  mPtRecoOverGen_GenEta_300_Inf_Cent_0_10 = ibooker.bookProfile(
      "PtRecoOverGen_GenEta_300_Inf_Cent_0_10", "300<genpt<Inf (0-10%);geneta;recopt/genpt", 90, etaRange, 0, 2, " ");

  mPtRecoOverGen_GenEta_20_30_Cent_10_30 = ibooker.bookProfile(
      "PtRecoOverGen_GenEta_20_30_Cent_10_30", "20<genpt<30 (10-30%);geneta;recopt/genpt", 90, etaRange, 0, 2, " ");
  mPtRecoOverGen_GenEta_30_50_Cent_10_30 = ibooker.bookProfile(
      "PtRecoOverGen_GenEta_30_50_Cent_10_30", "30<genpt<50 (10-30%);geneta;recopt/genpt", 90, etaRange, 0, 2, " ");
  mPtRecoOverGen_GenEta_50_80_Cent_10_30 = ibooker.bookProfile(
      "PtRecoOverGen_GenEta_50_80_Cent_10_30", "50<genpt<80 (10-30%);geneta;recopt/genpt", 90, etaRange, 0, 2, " ");
  mPtRecoOverGen_GenEta_80_120_Cent_10_30 = ibooker.bookProfile(
      "PtRecoOverGen_GenEta_80_120_Cent_10_30", "80<genpt<120 (10-30%);geneta;recopt/genpt", 90, etaRange, 0, 2, " ");
  mPtRecoOverGen_GenEta_120_180_Cent_10_30 = ibooker.bookProfile(
      "PtRecoOverGen_GenEta_120_180_Cent_10_30", "120<genpt<180 (10-30%);geneta;recopt/genpt", 90, etaRange, 0, 2, " ");
  mPtRecoOverGen_GenEta_180_300_Cent_10_30 = ibooker.bookProfile(
      "PtRecoOverGen_GenEta_180_300_Cent_10_30", "180<genpt<300 (10-30%);geneta;recopt/genpt", 90, etaRange, 0, 2, " ");
  mPtRecoOverGen_GenEta_300_Inf_Cent_10_30 = ibooker.bookProfile(
      "PtRecoOverGen_GenEta_300_Inf_Cent_10_30", "300<genpt<Inf (10-30%);geneta;recopt/genpt", 90, etaRange, 0, 2, " ");

  mPtRecoOverGen_GenEta_20_30_Cent_30_50 = ibooker.bookProfile(
      "PtRecoOverGen_GenEta_20_30_Cent_30_50", "20<genpt<30 (30-50%);geneta;recopt/genpt", 90, etaRange, 0, 2, " ");
  mPtRecoOverGen_GenEta_30_50_Cent_30_50 = ibooker.bookProfile(
      "PtRecoOverGen_GenEta_30_50_Cent_30_50", "30<genpt<50 (30-50%);geneta;recopt/genpt", 90, etaRange, 0, 2, " ");
  mPtRecoOverGen_GenEta_50_80_Cent_30_50 = ibooker.bookProfile(
      "PtRecoOverGen_GenEta_50_80_Cent_30_50", "50<genpt<80 (30-50%);geneta;recopt/genpt", 90, etaRange, 0, 2, " ");
  mPtRecoOverGen_GenEta_80_120_Cent_30_50 = ibooker.bookProfile(
      "PtRecoOverGen_GenEta_80_120_Cent_30_50", "80<genpt<120 (30-50%);geneta;recopt/genpt", 90, etaRange, 0, 2, " ");
  mPtRecoOverGen_GenEta_120_180_Cent_30_50 = ibooker.bookProfile(
      "PtRecoOverGen_GenEta_120_180_Cent_30_50", "120<genpt<180 (30-50%);geneta;recopt/genpt", 90, etaRange, 0, 2, " ");
  mPtRecoOverGen_GenEta_180_300_Cent_30_50 = ibooker.bookProfile(
      "PtRecoOverGen_GenEta_180_300_Cent_30_50", "180<genpt<300 (30-50%);geneta;recopt/genpt", 90, etaRange, 0, 2, " ");
  mPtRecoOverGen_GenEta_300_Inf_Cent_30_50 = ibooker.bookProfile(
      "PtRecoOverGen_GenEta_300_Inf_Cent_30_50", "300<genpt<Inf (30-50%);geneta;recopt/genpt", 90, etaRange, 0, 2, " ");

  mPtRecoOverGen_GenEta_20_30_Cent_50_80 = ibooker.bookProfile(
      "PtRecoOverGen_GenEta_20_30_Cent_50_80", "20<genpt<30 (50-80%);geneta;recopt/genpt", 90, etaRange, 0, 2, " ");
  mPtRecoOverGen_GenEta_30_50_Cent_50_80 = ibooker.bookProfile(
      "PtRecoOverGen_GenEta_30_50_Cent_50_80", "30<genpt<50 (50-80%);geneta;recopt/genpt", 90, etaRange, 0, 2, " ");
  mPtRecoOverGen_GenEta_50_80_Cent_50_80 = ibooker.bookProfile(
      "PtRecoOverGen_GenEta_50_80_Cent_50_80", "50<genpt<80 (50-80%);geneta;recopt/genpt", 90, etaRange, 0, 2, " ");
  mPtRecoOverGen_GenEta_80_120_Cent_50_80 = ibooker.bookProfile(
      "PtRecoOverGen_GenEta_80_120_Cent_50_80", "80<genpt<120 (50-80%);geneta;recopt/genpt", 90, etaRange, 0, 2, " ");
  mPtRecoOverGen_GenEta_120_180_Cent_50_80 = ibooker.bookProfile(
      "PtRecoOverGen_GenEta_120_180_Cent_50_80", "120<genpt<180 (50-80%);geneta;recopt/genpt", 90, etaRange, 0, 2, " ");
  mPtRecoOverGen_GenEta_180_300_Cent_50_80 = ibooker.bookProfile(
      "PtRecoOverGen_GenEta_180_300_Cent_50_80", "180<genpt<300 (50-80%);geneta;recopt/genpt", 90, etaRange, 0, 2, " ");
  mPtRecoOverGen_GenEta_300_Inf_Cent_50_80 = ibooker.bookProfile(
      "PtRecoOverGen_GenEta_300_Inf_Cent_50_80", "300<genpt<Inf (50-80%);geneta;recopt/genpt", 90, etaRange, 0, 2, " ");

  if (mOutputFile.empty())
    LogInfo("OutputInfo") << " Histograms will NOT be saved";
  else
    LogInfo("OutputInfo") << " Histograms will be saved to file:" << mOutputFile;

  delete h2D_etabins_vs_pt2;
  delete h2D_etabins_vs_pt;
  delete h2D_etabins_vs_phi;
  delete h2D_pfcand_etabins_vs_pt;
}

//------------------------------------------------------------------------------
// ~JetTester_HeavyIons
//------------------------------------------------------------------------------
JetTester_HeavyIons::~JetTester_HeavyIons() {}

//------------------------------------------------------------------------------
// beginJob
//------------------------------------------------------------------------------
// void JetTester_HeavyIons::beginJob() {
//  //std::cout<<"inside the begin job function"<<endl;
//}

//------------------------------------------------------------------------------
// endJob
//------------------------------------------------------------------------------
// void JetTester_HeavyIons::endJob()
//{
//  if (!mOutputFile.empty() && &*edm::Service<DQMStore>())
//    {
//      edm::Service<DQMStore>()->save(mOutputFile);
//    }
//}

//------------------------------------------------------------------------------
// analyze
//------------------------------------------------------------------------------
void JetTester_HeavyIons::analyze(const edm::Event &mEvent, const edm::EventSetup &mSetup) {
  // Get the primary vertices
  //----------------------------------------------------------------------------
  edm::Handle<vector<reco::Vertex>> pvHandle;
  mEvent.getByToken(pvToken_, pvHandle);
  reco::Vertex::Point vtx(0, 0, 0);
  edm::Handle<reco::VertexCollection> vtxs;
  // vtx = getVtx(mEvent);

  mEvent.getByToken(hiVertexToken_, vtxs);
  int greatestvtx = 0;
  int nVertex = vtxs->size();

  for (unsigned int i = 0; i < vtxs->size(); ++i) {
    unsigned int daughter = (*vtxs)[i].tracksSize();
    if (daughter > (*vtxs)[greatestvtx].tracksSize())
      greatestvtx = i;
  }

  if (nVertex <= 0) {
    vtx = reco::Vertex::Point(0, 0, 0);
  }
  vtx = (*vtxs)[greatestvtx].position();

  int nGoodVertices = 0;

  if (pvHandle.isValid()) {
    for (unsigned i = 0; i < pvHandle->size(); i++) {
      if ((*pvHandle)[i].ndof() > 4 && (fabs((*pvHandle)[i].z()) <= 24) && (fabs((*pvHandle)[i].position().rho()) <= 2))
        nGoodVertices++;
    }
  }

  mNvtx->Fill(nGoodVertices);

  // Get the Jet collection
  //----------------------------------------------------------------------------
  // math::XYZTLorentzVector p4tmp[2];

  std::vector<Jet> recoJets;
  recoJets.clear();

  edm::Handle<CaloJetCollection> caloJets;
  edm::Handle<JPTJetCollection> jptJets;
  edm::Handle<PFJetCollection> pfJets;
  edm::Handle<BasicJetCollection> basicJets;

  // Get the Particle flow candidates and the Voronoi variables
  edm::Handle<reco::PFCandidateCollection> pfCandidates;
  edm::Handle<CaloTowerCollection> caloCandidates;
  edm::Handle<reco::CandidateView> pfcandidates_;
  edm::Handle<reco::CandidateView> calocandidates_;

  // get the centrality
  edm::Handle<reco::Centrality> cent;
  mEvent.getByToken(centralityToken, cent);  //_centralitytag comes from the cfg

  mHF->Fill(cent->EtHFtowerSum());
  Float_t HF_energy = cent->EtHFtowerSum();

  edm::Handle<int> cbin;
  mEvent.getByToken(centralityBinToken, cbin);
  if (!cent.isValid())
    return;

  int hibin = -999;
  if (cbin.isValid()) {
    hibin = *cbin;
  }
  // else  edm::LogWarning("JetTester_HeavyIons") << "invalid collection:
  // centralityBin " << std::endl;

  bool isCentral = false;
  bool ismidCentral = false;
  bool ismidPeripheral = false;
  bool isPeripheral = false;

  if (hibin < 20)
    isCentral = true;
  if (hibin >= 20 && hibin < 60)
    ismidCentral = true;
  if (hibin >= 60 && hibin < 100)
    ismidPeripheral = true;
  if (hibin >= 100 && hibin < 160)
    isPeripheral = true;

  if (isCaloJet)
    mEvent.getByToken(caloJetsToken_, caloJets);
  if (isJPTJet)
    mEvent.getByToken(jptJetsToken_, jptJets);
  if (isPFJet) {
    if (std::string("Pu") == UEAlgo)
      mEvent.getByToken(basicJetsToken_, basicJets);
  }

  mEvent.getByToken(pfCandToken_, pfCandidates);
  mEvent.getByToken(pfCandViewToken_, pfcandidates_);

  mEvent.getByToken(caloTowersToken_, caloCandidates);
  mEvent.getByToken(caloCandViewToken_, calocandidates_);

  const reco::PFCandidateCollection *pfCandidateColl = pfCandidates.product();

  Int_t NPFpart = 0;
  Int_t NCaloTower = 0;
  Float_t pfPt = 0;
  Float_t pfEta = 0;
  Int_t pfID = 0;
  Float_t pfPhi = 0;
  Float_t caloPt = 0;
  Float_t caloEta = 0;
  Float_t caloPhi = 0;
  Float_t SumPt_value = 0;

  double edge_pseudorapidity[etaBins_ + 1] = {-5.191,
                                              -2.650,
                                              -2.043,
                                              -1.740,
                                              -1.479,
                                              -1.131,
                                              -0.783,
                                              -0.522,
                                              0.522,
                                              0.783,
                                              1.131,
                                              1.479,
                                              1.740,
                                              2.043,
                                              2.650,
                                              5.191};

  if (isCaloJet) {
    Float_t SumCaloPt[etaBins_];
    Float_t SumSquaredCaloPt[etaBins_];

    for (int i = 0; i < etaBins_; i++) {
      SumCaloPt[i] = 0;
      SumSquaredCaloPt[i] = 0;
    }

    for (unsigned icand = 0; icand < caloCandidates->size(); icand++) {
      const CaloTower &tower = (*caloCandidates)[icand];
      reco::CandidateViewRef ref(calocandidates_, icand);
      if (tower.p4(vtx).Et() < 0.1)
        continue;

      NCaloTower++;

      caloPt = tower.p4(vtx).Et();
      caloEta = tower.p4(vtx).Eta();
      caloPhi = tower.p4(vtx).Phi();

      for (size_t k = 0; k < nedge_pseudorapidity - 1; k++) {
        if (caloEta >= edge_pseudorapidity[k] && caloEta < edge_pseudorapidity[k + 1]) {
          SumCaloPt[k] = SumCaloPt[k] + caloPt;
          SumSquaredCaloPt[k] = SumSquaredCaloPt[k] + caloPt * caloPt;
        }  // eta selection statement

      }  // eta bin loop

      SumPt_value = SumPt_value + caloPt;

      mCaloPt->Fill(caloPt);
      mCaloEta->Fill(caloEta);
      mCaloPhi->Fill(caloPhi);

    }  // calo tower candidate  loop

    Float_t Evt_SumCaloPt = 0;

    Float_t Evt_SumSquaredCaloPt = 0;

    mSumCaloPt_n5p191_n2p650->Fill(SumCaloPt[0]);
    mSumCaloPt_n2p650_n2p043->Fill(SumCaloPt[1]);
    mSumCaloPt_n2p043_n1p740->Fill(SumCaloPt[2]);
    mSumCaloPt_n1p740_n1p479->Fill(SumCaloPt[3]);
    mSumCaloPt_n1p479_n1p131->Fill(SumCaloPt[4]);
    mSumCaloPt_n1p131_n0p783->Fill(SumCaloPt[5]);
    mSumCaloPt_n0p783_n0p522->Fill(SumCaloPt[6]);
    mSumCaloPt_n0p522_0p522->Fill(SumCaloPt[7]);
    mSumCaloPt_0p522_0p783->Fill(SumCaloPt[8]);
    mSumCaloPt_0p783_1p131->Fill(SumCaloPt[9]);
    mSumCaloPt_1p131_1p479->Fill(SumCaloPt[10]);
    mSumCaloPt_1p479_1p740->Fill(SumCaloPt[11]);
    mSumCaloPt_1p740_2p043->Fill(SumCaloPt[12]);
    mSumCaloPt_2p043_2p650->Fill(SumCaloPt[13]);
    mSumCaloPt_2p650_5p191->Fill(SumCaloPt[14]);

    for (size_t k = 0; k < nedge_pseudorapidity - 1; k++) {
      Evt_SumCaloPt = Evt_SumCaloPt + SumCaloPt[k];

      Evt_SumSquaredCaloPt = Evt_SumSquaredCaloPt + SumSquaredCaloPt[k];

    }  // eta bin loop

    mSumCaloPt->Fill(Evt_SumCaloPt);

    mSumSquaredCaloPt->Fill(Evt_SumSquaredCaloPt);

    mSumCaloPt_HF->Fill(Evt_SumCaloPt, HF_energy);

    mNCalopart->Fill(NCaloTower);
    mSumpt->Fill(SumPt_value);

  }  // is calo jet

  if (isPFJet) {
    Float_t SumPFPt[etaBins_];

    Float_t SumSquaredPFPt[etaBins_];

    for (int i = 0; i < etaBins_; i++) {
      SumPFPt[i] = 0;
      SumSquaredPFPt[i] = 0;
    }

    for (unsigned icand = 0; icand < pfCandidateColl->size(); icand++) {
      const reco::PFCandidate pfCandidate = pfCandidateColl->at(icand);
      reco::CandidateViewRef ref(pfcandidates_, icand);

      if (pfCandidate.pt() < 0.5)
        continue;

      NPFpart++;
      pfPt = pfCandidate.pt();
      pfEta = pfCandidate.eta();
      pfPhi = pfCandidate.phi();
      pfID = pfCandidate.particleId();

      bool inBarrel = false;
      bool inEndcap = false;
      bool inForward = false;

      if (fabs(pfEta) < BarrelEta)
        inBarrel = true;
      if (fabs(pfEta) >= BarrelEta && fabs(pfEta) < EndcapEta)
        inEndcap = true;
      if (fabs(pfEta) >= EndcapEta && fabs(pfEta) < ForwardEta)
        inForward = true;

      switch (pfID) {
        case 0:
          mPFCandpT_vs_eta_Unknown->Fill(pfPt, pfEta);
          if (inBarrel)
            mPFCandpT_Barrel_Unknown->Fill(pfPt);
          if (inEndcap)
            mPFCandpT_Endcap_Unknown->Fill(pfPt);
          if (inForward)
            mPFCandpT_Forward_Unknown->Fill(pfPt);
        case 1:
          mPFCandpT_vs_eta_ChargedHadron->Fill(pfPt, pfEta);
          if (inBarrel)
            mPFCandpT_Barrel_ChargedHadron->Fill(pfPt);
          if (inEndcap)
            mPFCandpT_Endcap_ChargedHadron->Fill(pfPt);
          if (inForward)
            mPFCandpT_Forward_ChargedHadron->Fill(pfPt);
        case 2:
          mPFCandpT_vs_eta_electron->Fill(pfPt, pfEta);
          if (inBarrel)
            mPFCandpT_Barrel_electron->Fill(pfPt);
          if (inEndcap)
            mPFCandpT_Endcap_electron->Fill(pfPt);
          if (inForward)
            mPFCandpT_Forward_electron->Fill(pfPt);
        case 3:
          mPFCandpT_vs_eta_muon->Fill(pfPt, pfEta);
          if (inBarrel)
            mPFCandpT_Barrel_muon->Fill(pfPt);
          if (inEndcap)
            mPFCandpT_Endcap_muon->Fill(pfPt);
          if (inForward)
            mPFCandpT_Forward_muon->Fill(pfPt);
        case 4:
          mPFCandpT_vs_eta_photon->Fill(pfPt, pfEta);
          if (inBarrel)
            mPFCandpT_Barrel_photon->Fill(pfPt);
          if (inEndcap)
            mPFCandpT_Endcap_photon->Fill(pfPt);
          if (inForward)
            mPFCandpT_Forward_photon->Fill(pfPt);
        case 5:
          mPFCandpT_vs_eta_NeutralHadron->Fill(pfPt, pfEta);
          if (inBarrel)
            mPFCandpT_Barrel_NeutralHadron->Fill(pfPt);
          if (inEndcap)
            mPFCandpT_Endcap_NeutralHadron->Fill(pfPt);
          if (inForward)
            mPFCandpT_Forward_NeutralHadron->Fill(pfPt);
        case 6:
          mPFCandpT_vs_eta_HadE_inHF->Fill(pfPt, pfEta);
          if (inBarrel)
            mPFCandpT_Barrel_HadE_inHF->Fill(pfPt);
          if (inEndcap)
            mPFCandpT_Endcap_HadE_inHF->Fill(pfPt);
          if (inForward)
            mPFCandpT_Forward_HadE_inHF->Fill(pfPt);
        case 7:
          mPFCandpT_vs_eta_EME_inHF->Fill(pfPt, pfEta);
          if (inBarrel)
            mPFCandpT_Barrel_EME_inHF->Fill(pfPt);
          if (inEndcap)
            mPFCandpT_Endcap_EME_inHF->Fill(pfPt);
          if (inForward)
            mPFCandpT_Forward_EME_inHF->Fill(pfPt);
      }

      for (size_t k = 0; k < nedge_pseudorapidity - 1; k++) {
        if (pfEta >= edge_pseudorapidity[k] && pfEta < edge_pseudorapidity[k + 1]) {
          SumPFPt[k] = SumPFPt[k] + pfPt;

          SumSquaredPFPt[k] = SumSquaredPFPt[k] + pfPt * pfPt;

        }  // eta selection statement

      }  // eta bin loop

      SumPt_value = SumPt_value + pfPt;

      mPFPt->Fill(pfPt);
      mPFEta->Fill(pfEta);
      mPFPhi->Fill(pfPhi);

    }  // pf candidate loop

    Float_t Evt_SumPFPt = 0;

    Float_t Evt_SumSquaredPFPt = 0;

    mSumPFPt_n5p191_n2p650->Fill(SumPFPt[0]);
    mSumPFPt_n2p650_n2p043->Fill(SumPFPt[1]);
    mSumPFPt_n2p043_n1p740->Fill(SumPFPt[2]);
    mSumPFPt_n1p740_n1p479->Fill(SumPFPt[3]);
    mSumPFPt_n1p479_n1p131->Fill(SumPFPt[4]);
    mSumPFPt_n1p131_n0p783->Fill(SumPFPt[5]);
    mSumPFPt_n0p783_n0p522->Fill(SumPFPt[6]);
    mSumPFPt_n0p522_0p522->Fill(SumPFPt[7]);
    mSumPFPt_0p522_0p783->Fill(SumPFPt[8]);
    mSumPFPt_0p783_1p131->Fill(SumPFPt[9]);
    mSumPFPt_1p131_1p479->Fill(SumPFPt[10]);
    mSumPFPt_1p479_1p740->Fill(SumPFPt[11]);
    mSumPFPt_1p740_2p043->Fill(SumPFPt[12]);
    mSumPFPt_2p043_2p650->Fill(SumPFPt[13]);
    mSumPFPt_2p650_5p191->Fill(SumPFPt[14]);

    for (size_t k = 0; k < nedge_pseudorapidity - 1; k++) {
      Evt_SumPFPt = Evt_SumPFPt + SumPFPt[k];

      Evt_SumSquaredPFPt = Evt_SumSquaredPFPt + SumSquaredPFPt[k];

    }  // eta bin loop

    mSumPFPt->Fill(Evt_SumPFPt);

    mSumSquaredPFPt->Fill(Evt_SumSquaredPFPt);

    mSumPFPt_HF->Fill(Evt_SumPFPt, HF_energy);

    mNPFpart->Fill(NPFpart);
    mSumpt->Fill(SumPt_value);
  }

  if (isCaloJet) {
    for (unsigned ijet = 0; ijet < caloJets->size(); ijet++)
      recoJets.push_back((*caloJets)[ijet]);
  }

  if (isJPTJet) {
    for (unsigned ijet = 0; ijet < jptJets->size(); ijet++)
      recoJets.push_back((*jptJets)[ijet]);
  }

  if (isPFJet) {
    if (std::string("Pu") == UEAlgo) {
      for (unsigned ijet = 0; ijet < basicJets->size(); ijet++)
        recoJets.push_back((*basicJets)[ijet]);
    }
  }

  if (isCaloJet && !caloJets.isValid())
    return;
  if (isJPTJet && !jptJets.isValid())
    return;
  if (isPFJet) {
    if (std::string("Pu") == UEAlgo) {
      if (!basicJets.isValid())
        return;
    }
  }

  int nJet_40 = 0;

  mNJets->Fill(recoJets.size());

  for (unsigned ijet = 0; ijet < recoJets.size(); ijet++) {
    if (recoJets[ijet].pt() > mRecoJetPtThreshold) {
      // counting forward and barrel jets
      // get an idea of no of jets with pT>40 GeV
      if (recoJets[ijet].pt() > 40)
        nJet_40++;

      if (mEta)
        mEta->Fill(recoJets[ijet].eta());
      if (mjetpileup)
        mjetpileup->Fill(recoJets[ijet].pileup());
      if (mJetArea)
        mJetArea->Fill(recoJets[ijet].jetArea());
      if (mPhi)
        mPhi->Fill(recoJets[ijet].phi());
      if (mEnergy)
        mEnergy->Fill(recoJets[ijet].energy());
      if (mP)
        mP->Fill(recoJets[ijet].p());
      if (mPt)
        mPt->Fill(recoJets[ijet].pt());
      if (mMass)
        mMass->Fill(recoJets[ijet].mass());
      if (mConstituents)
        mConstituents->Fill(recoJets[ijet].nConstituents());
    }
  }

  if (mNJets_40)
    mNJets_40->Fill(nJet_40);

  // Gen level information:
  if (!mEvent.isRealData()) {
    // Get ptHat
    //------------------------------------------------------------------------
    edm::Handle<GenEventInfoProduct> myGenEvt;
    mEvent.getByToken(evtToken_, myGenEvt);

    if (myGenEvt.isValid()) {
      if (myGenEvt->hasBinningValues()) {
        double ptHat = myGenEvt->binningValues()[0];
        if (mPtHat)
          mPtHat->Fill(ptHat);
      }
    }
    // Gen jets
    //------------------------------------------------------------------------
    edm::Handle<GenJetCollection> genJets;
    mEvent.getByToken(genJetsToken_, genJets);

    if (!genJets.isValid())
      return;

    for (GenJetCollection::const_iterator gjet = genJets->begin(); gjet != genJets->end(); gjet++) {
      if (gjet->pt() > mMatchGenPtThreshold) {
        if (mGenEta)
          mGenEta->Fill(gjet->eta());
        if (mGenPhi)
          mGenPhi->Fill(gjet->phi());
        if (mGenPt)
          mGenPt->Fill(gjet->pt());
      }
    }

    if (!(mInputGenCollection.label().empty())) {
      for (GenJetCollection::const_iterator gjet = genJets->begin(); gjet != genJets->end(); gjet++) {
        if (fabs(gjet->eta()) > 6.)
          continue;  // Out of the detector
        if (gjet->pt() < mMatchGenPtThreshold)
          continue;
        if (recoJets.empty())
          continue;

        bool inBarrel = false;
        bool inEndcap = false;
        bool inForward = false;

        if (fabs(gjet->eta()) < BarrelEta)
          inBarrel = true;
        if (fabs(gjet->eta()) >= BarrelEta && fabs(gjet->eta()) < EndcapEta)
          inEndcap = true;
        if (fabs(gjet->eta()) >= EndcapEta && fabs(gjet->eta()) < ForwardEta)
          inForward = true;

        // pt response
        //------------------------------------------------------------
        int iMatch = -1;
        double deltaRBest = 999;
        double JetPtBest = 0;
        for (unsigned ijet = 0; ijet < recoJets.size(); ++ijet) {
          double recoPt = recoJets[ijet].pt();
          if (recoPt > 10) {
            double delR = deltaR(gjet->eta(), gjet->phi(), recoJets[ijet].eta(), recoJets[ijet].phi());
            if (delR < deltaRBest) {
              deltaRBest = delR;
              JetPtBest = recoPt;
              iMatch = ijet;
            }
          }
        }
        if (iMatch < 0)
          continue;

        // fillMatchHists(gjet->eta(),  gjet->phi(),  gjet->pt(),
        // recoJets[iMatch].eta(), recoJets[iMatch].phi(),
        // recoJets[iMatch].pt(), hibin);
        if (deltaRBest < mRThreshold) {
          double genpt = gjet->pt();
          double geneta = gjet->eta();
          double response = JetPtBest / genpt;
          // Fill all the response histograms here: for each pT bin, eta region,
          // centrality bin

          if (inBarrel) {
            if (isCentral)
              mPtRecoOverGen_GenPt_B_Cent_0_10->Fill(log10(genpt), response);
            if (ismidCentral)
              mPtRecoOverGen_GenPt_B_Cent_10_30->Fill(log10(genpt), response);
            if (ismidPeripheral)
              mPtRecoOverGen_GenPt_B_Cent_30_50->Fill(log10(genpt), response);
            if (isPeripheral)
              mPtRecoOverGen_GenPt_B_Cent_50_80->Fill(log10(genpt), response);
          }
          if (inEndcap) {
            if (isCentral)
              mPtRecoOverGen_GenPt_E_Cent_0_10->Fill(log10(genpt), response);
            if (ismidCentral)
              mPtRecoOverGen_GenPt_E_Cent_10_30->Fill(log10(genpt), response);
            if (ismidPeripheral)
              mPtRecoOverGen_GenPt_E_Cent_30_50->Fill(log10(genpt), response);
            if (isPeripheral)
              mPtRecoOverGen_GenPt_E_Cent_50_80->Fill(log10(genpt), response);
          }
          if (inForward) {
            if (isCentral)
              mPtRecoOverGen_GenPt_F_Cent_0_10->Fill(log10(genpt), response);
            if (ismidCentral)
              mPtRecoOverGen_GenPt_F_Cent_10_30->Fill(log10(genpt), response);
            if (ismidPeripheral)
              mPtRecoOverGen_GenPt_F_Cent_30_50->Fill(log10(genpt), response);
            if (isPeripheral)
              mPtRecoOverGen_GenPt_F_Cent_50_80->Fill(log10(genpt), response);
          }

          if (gjet->pt() >= 20 && gjet->pt() < 30) {
            if (isCentral) {
              mPtRecoOverGen_GenEta_20_30_Cent_0_10->Fill(geneta, response);
              if (inBarrel)
                mPtRecoOverGen_B_20_30_Cent_0_10->Fill(response);
              if (inEndcap)
                mPtRecoOverGen_E_20_30_Cent_0_10->Fill(response);
              if (inForward)
                mPtRecoOverGen_F_20_30_Cent_0_10->Fill(response);
            }  //
            if (ismidCentral) {
              mPtRecoOverGen_GenEta_20_30_Cent_10_30->Fill(geneta, response);
              if (inBarrel)
                mPtRecoOverGen_B_20_30_Cent_10_30->Fill(response);
              if (inEndcap)
                mPtRecoOverGen_E_20_30_Cent_10_30->Fill(response);
              if (inForward)
                mPtRecoOverGen_F_20_30_Cent_10_30->Fill(response);
            }  //
            if (ismidPeripheral) {
              mPtRecoOverGen_GenEta_20_30_Cent_30_50->Fill(geneta, response);
              if (inBarrel)
                mPtRecoOverGen_B_20_30_Cent_30_50->Fill(response);
              if (inEndcap)
                mPtRecoOverGen_E_20_30_Cent_30_50->Fill(response);
              if (inForward)
                mPtRecoOverGen_F_20_30_Cent_30_50->Fill(response);
            }  //
            if (isPeripheral) {
              mPtRecoOverGen_GenEta_20_30_Cent_50_80->Fill(geneta, response);
              if (inBarrel)
                mPtRecoOverGen_B_20_30_Cent_50_80->Fill(response);
              if (inEndcap)
                mPtRecoOverGen_E_20_30_Cent_50_80->Fill(response);
              if (inForward)
                mPtRecoOverGen_F_20_30_Cent_50_80->Fill(response);
            }  //
          }    // pt bin 20-30

          if (gjet->pt() >= 30 && gjet->pt() < 50) {
            if (isCentral) {
              mPtRecoOverGen_GenEta_30_50_Cent_0_10->Fill(geneta, response);
              if (inBarrel)
                mPtRecoOverGen_B_30_50_Cent_0_10->Fill(response);
              if (inEndcap)
                mPtRecoOverGen_E_30_50_Cent_0_10->Fill(response);
              if (inForward)
                mPtRecoOverGen_F_30_50_Cent_0_10->Fill(response);
            }  //
            if (ismidCentral) {
              mPtRecoOverGen_GenEta_30_50_Cent_10_30->Fill(geneta, response);
              if (inBarrel)
                mPtRecoOverGen_B_30_50_Cent_10_30->Fill(response);
              if (inEndcap)
                mPtRecoOverGen_E_30_50_Cent_10_30->Fill(response);
              if (inForward)
                mPtRecoOverGen_F_30_50_Cent_10_30->Fill(response);
            }  //
            if (ismidPeripheral) {
              mPtRecoOverGen_GenEta_30_50_Cent_30_50->Fill(geneta, response);
              if (inBarrel)
                mPtRecoOverGen_B_30_50_Cent_30_50->Fill(response);
              if (inEndcap)
                mPtRecoOverGen_E_30_50_Cent_30_50->Fill(response);
              if (inForward)
                mPtRecoOverGen_F_30_50_Cent_30_50->Fill(response);
            }  //
            if (isPeripheral) {
              mPtRecoOverGen_GenEta_30_50_Cent_50_80->Fill(geneta, response);
              if (inBarrel)
                mPtRecoOverGen_B_30_50_Cent_50_80->Fill(response);
              if (inEndcap)
                mPtRecoOverGen_E_30_50_Cent_50_80->Fill(response);
              if (inForward)
                mPtRecoOverGen_F_30_50_Cent_50_80->Fill(response);
            }  //
          }    // pt bin 30-50

          if (gjet->pt() >= 50 && gjet->pt() < 80) {
            if (isCentral) {
              mPtRecoOverGen_GenEta_50_80_Cent_0_10->Fill(geneta, response);
              if (inBarrel)
                mPtRecoOverGen_B_50_80_Cent_0_10->Fill(response);
              if (inEndcap)
                mPtRecoOverGen_E_50_80_Cent_0_10->Fill(response);
              if (inForward)
                mPtRecoOverGen_F_50_80_Cent_0_10->Fill(response);
            }  //
            if (ismidCentral) {
              mPtRecoOverGen_GenEta_50_80_Cent_10_30->Fill(geneta, response);
              if (inBarrel)
                mPtRecoOverGen_B_50_80_Cent_10_30->Fill(response);
              if (inEndcap)
                mPtRecoOverGen_E_50_80_Cent_10_30->Fill(response);
              if (inForward)
                mPtRecoOverGen_F_50_80_Cent_10_30->Fill(response);
            }  //
            if (ismidPeripheral) {
              mPtRecoOverGen_GenEta_50_80_Cent_30_50->Fill(geneta, response);
              if (inBarrel)
                mPtRecoOverGen_B_50_80_Cent_30_50->Fill(response);
              if (inEndcap)
                mPtRecoOverGen_E_50_80_Cent_30_50->Fill(response);
              if (inForward)
                mPtRecoOverGen_F_50_80_Cent_30_50->Fill(response);
            }  //
            if (isPeripheral) {
              mPtRecoOverGen_GenEta_50_80_Cent_50_80->Fill(geneta, response);
              if (inBarrel)
                mPtRecoOverGen_B_50_80_Cent_50_80->Fill(response);
              if (inEndcap)
                mPtRecoOverGen_E_50_80_Cent_50_80->Fill(response);
              if (inForward)
                mPtRecoOverGen_F_50_80_Cent_50_80->Fill(response);
            }  //
          }    // pt bin 50-80

          if (gjet->pt() >= 80 && gjet->pt() < 120) {
            if (isCentral) {
              mPtRecoOverGen_GenEta_80_120_Cent_0_10->Fill(geneta, response);
              if (inBarrel)
                mPtRecoOverGen_B_80_120_Cent_0_10->Fill(response);
              if (inEndcap)
                mPtRecoOverGen_E_80_120_Cent_0_10->Fill(response);
              if (inForward)
                mPtRecoOverGen_F_80_120_Cent_0_10->Fill(response);
            }  //
            if (ismidCentral) {
              mPtRecoOverGen_GenEta_80_120_Cent_10_30->Fill(geneta, response);
              if (inBarrel)
                mPtRecoOverGen_B_80_120_Cent_10_30->Fill(response);
              if (inEndcap)
                mPtRecoOverGen_E_80_120_Cent_10_30->Fill(response);
              if (inForward)
                mPtRecoOverGen_F_80_120_Cent_10_30->Fill(response);
            }  //
            if (ismidPeripheral) {
              mPtRecoOverGen_GenEta_80_120_Cent_30_50->Fill(geneta, response);
              if (inBarrel)
                mPtRecoOverGen_B_80_120_Cent_30_50->Fill(response);
              if (inEndcap)
                mPtRecoOverGen_E_80_120_Cent_30_50->Fill(response);
              if (inForward)
                mPtRecoOverGen_F_80_120_Cent_30_50->Fill(response);
            }  //
            if (isPeripheral) {
              mPtRecoOverGen_GenEta_80_120_Cent_50_80->Fill(geneta, response);
              if (inBarrel)
                mPtRecoOverGen_B_80_120_Cent_50_80->Fill(response);
              if (inEndcap)
                mPtRecoOverGen_E_80_120_Cent_50_80->Fill(response);
              if (inForward)
                mPtRecoOverGen_F_80_120_Cent_50_80->Fill(response);
            }  //
          }    // pt bin 80-120

          if (gjet->pt() >= 120 && gjet->pt() < 180) {
            if (isCentral) {
              mPtRecoOverGen_GenEta_120_180_Cent_0_10->Fill(geneta, response);
              if (inBarrel)
                mPtRecoOverGen_B_120_180_Cent_0_10->Fill(response);
              if (inEndcap)
                mPtRecoOverGen_E_120_180_Cent_0_10->Fill(response);
              if (inForward)
                mPtRecoOverGen_F_120_180_Cent_0_10->Fill(response);
            }  //
            if (ismidCentral) {
              mPtRecoOverGen_GenEta_120_180_Cent_10_30->Fill(geneta, response);
              if (inBarrel)
                mPtRecoOverGen_B_120_180_Cent_10_30->Fill(response);
              if (inEndcap)
                mPtRecoOverGen_E_120_180_Cent_10_30->Fill(response);
              if (inForward)
                mPtRecoOverGen_F_120_180_Cent_10_30->Fill(response);
            }  //
            if (ismidPeripheral) {
              mPtRecoOverGen_GenEta_120_180_Cent_30_50->Fill(geneta, response);
              if (inBarrel)
                mPtRecoOverGen_B_120_180_Cent_30_50->Fill(response);
              if (inEndcap)
                mPtRecoOverGen_E_120_180_Cent_30_50->Fill(response);
              if (inForward)
                mPtRecoOverGen_F_120_180_Cent_30_50->Fill(response);
            }  //
            if (isPeripheral) {
              mPtRecoOverGen_GenEta_120_180_Cent_50_80->Fill(geneta, response);
              if (inBarrel)
                mPtRecoOverGen_B_120_180_Cent_50_80->Fill(response);
              if (inEndcap)
                mPtRecoOverGen_E_120_180_Cent_50_80->Fill(response);
              if (inForward)
                mPtRecoOverGen_F_120_180_Cent_50_80->Fill(response);
            }  //
          }    // pt bin 120-180

          if (gjet->pt() >= 180 && gjet->pt() < 300) {
            if (isCentral) {
              mPtRecoOverGen_GenEta_180_300_Cent_0_10->Fill(geneta, response);
              if (inBarrel)
                mPtRecoOverGen_B_180_300_Cent_0_10->Fill(response);
              if (inEndcap)
                mPtRecoOverGen_E_180_300_Cent_0_10->Fill(response);
              if (inForward)
                mPtRecoOverGen_F_180_300_Cent_0_10->Fill(response);
            }  //
            if (ismidCentral) {
              mPtRecoOverGen_GenEta_180_300_Cent_10_30->Fill(geneta, response);
              if (inBarrel)
                mPtRecoOverGen_B_180_300_Cent_10_30->Fill(response);
              if (inEndcap)
                mPtRecoOverGen_E_180_300_Cent_10_30->Fill(response);
              if (inForward)
                mPtRecoOverGen_F_180_300_Cent_10_30->Fill(response);
            }  //
            if (ismidPeripheral) {
              mPtRecoOverGen_GenEta_180_300_Cent_30_50->Fill(geneta, response);
              if (inBarrel)
                mPtRecoOverGen_B_180_300_Cent_30_50->Fill(response);
              if (inEndcap)
                mPtRecoOverGen_E_180_300_Cent_30_50->Fill(response);
              if (inForward)
                mPtRecoOverGen_F_180_300_Cent_30_50->Fill(response);
            }  //
            if (isPeripheral) {
              mPtRecoOverGen_GenEta_180_300_Cent_50_80->Fill(geneta, response);
              if (inBarrel)
                mPtRecoOverGen_B_180_300_Cent_50_80->Fill(response);
              if (inEndcap)
                mPtRecoOverGen_E_180_300_Cent_50_80->Fill(response);
              if (inForward)
                mPtRecoOverGen_F_180_300_Cent_50_80->Fill(response);
            }  //
          }    // pt bin 180-300

          if (gjet->pt() >= 300) {
            if (isCentral) {
              mPtRecoOverGen_GenEta_300_Inf_Cent_0_10->Fill(geneta, response);
              if (inBarrel)
                mPtRecoOverGen_B_300_Inf_Cent_0_10->Fill(response);
              if (inEndcap)
                mPtRecoOverGen_E_300_Inf_Cent_0_10->Fill(response);
              if (inForward)
                mPtRecoOverGen_F_300_Inf_Cent_0_10->Fill(response);
            }  //
            if (ismidCentral) {
              mPtRecoOverGen_GenEta_300_Inf_Cent_10_30->Fill(geneta, response);
              if (inBarrel)
                mPtRecoOverGen_B_300_Inf_Cent_10_30->Fill(response);
              if (inEndcap)
                mPtRecoOverGen_E_300_Inf_Cent_10_30->Fill(response);
              if (inForward)
                mPtRecoOverGen_F_300_Inf_Cent_10_30->Fill(response);
            }  //
            if (ismidPeripheral) {
              mPtRecoOverGen_GenEta_300_Inf_Cent_30_50->Fill(geneta, response);
              if (inBarrel)
                mPtRecoOverGen_B_300_Inf_Cent_30_50->Fill(response);
              if (inEndcap)
                mPtRecoOverGen_E_300_Inf_Cent_30_50->Fill(response);
              if (inForward)
                mPtRecoOverGen_F_300_Inf_Cent_30_50->Fill(response);
            }  //
            if (isPeripheral) {
              mPtRecoOverGen_GenEta_300_Inf_Cent_50_80->Fill(geneta, response);
              if (inBarrel)
                mPtRecoOverGen_B_300_Inf_Cent_50_80->Fill(response);
              if (inEndcap)
                mPtRecoOverGen_E_300_Inf_Cent_50_80->Fill(response);
              if (inForward)
                mPtRecoOverGen_F_300_Inf_Cent_50_80->Fill(response);
            }  //
          }    // pt bin 300-Inf

        }  // delta R < mRthreshold

      }  // gen jet collection loop

    }  // not empty gen collection

  }  // is the event real
}
