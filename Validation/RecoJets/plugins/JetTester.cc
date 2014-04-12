// Producer for validation histograms for Calo, JPT and PF jet objects
// F. Ratnikov, Sept. 7, 2006
// Modified by Chiyoung Jeong, Feb. 2, 2010
// Modified by J. Piedra, Sept. 11, 2013

#include "JetTester.h"

using namespace edm;
using namespace reco;
using namespace std;

JetTester::JetTester(const edm::ParameterSet& iConfig) :
  mInputCollection               (iConfig.getParameter<edm::InputTag>       ("src")),
  mInputGenCollection            (iConfig.getParameter<edm::InputTag>       ("srcGen")),
//  rhoTag                         (iConfig.getParameter<edm::InputTag>       ("srcRho")), 
  mOutputFile                    (iConfig.getUntrackedParameter<std::string>("OutputFile","")),
  JetType                        (iConfig.getUntrackedParameter<std::string>("JetType")),
  mRecoJetPtThreshold            (iConfig.getParameter<double>              ("recoJetPtThreshold")),
  mMatchGenPtThreshold           (iConfig.getParameter<double>              ("matchGenPtThreshold")),
  mGenEnergyFractionThreshold    (iConfig.getParameter<double>              ("genEnergyFractionThreshold")),
  mRThreshold                    (iConfig.getParameter<double>              ("RThreshold")),
  JetCorrectionService           (iConfig.getParameter<std::string>         ("JetCorrections"))
{
  std::string inputCollectionLabel(mInputCollection.label());

//  std::size_t foundCaloCollection = inputCollectionLabel.find("Calo");
//  std::size_t foundJPTCollection  = inputCollectionLabel.find("JetPlusTrack");
//  std::size_t foundPFCollection   = inputCollectionLabel.find("PF");

  isCaloJet = (std::string("calo")==JetType);
  isJPTJet  = (std::string("jpt") ==JetType);
  isPFJet   = (std::string("pf")  ==JetType);

  //consumes
   pvToken_ = consumes<std::vector<reco::Vertex> >(edm::InputTag("offlinePrimaryVertices"));
   caloTowersToken_ = consumes<CaloTowerCollection>(edm::InputTag("towerMaker"));
   if (isCaloJet) caloJetsToken_  = consumes<reco::CaloJetCollection>(mInputCollection);
   if (isJPTJet)  jptJetsToken_   = consumes<reco::JPTJetCollection>(mInputCollection);
   if (isPFJet)   pfJetsToken_    = consumes<reco::PFJetCollection>(mInputCollection);
   genJetsToken_ = consumes<reco::GenJetCollection>(edm::InputTag(mInputGenCollection));
   evtToken_ = consumes<edm::HepMCProduct>(edm::InputTag("generator"));


  // Events variables
  mNvtx           = 0;

  // Jet parameters
  mEta          = 0;
  mPhi          = 0;
  mEnergy       = 0;
  mP            = 0;
  mPt           = 0;
  mMass         = 0;
  mConstituents = 0;
  mHadTiming    = 0;
  mEmTiming     = 0;
  mJetArea      = 0;
//  mRho          = 0;

  // Corrected jets
  mCorrJetPt  = 0;
  mCorrJetEta = 0;
  mCorrJetPhi = 0;
  mCorrJetEta_Pt40 = 0;
  mCorrJetPhi_Pt40 = 0;

  // Corrected jets profiles
  mPtCorrOverReco_Pt_B                = 0;
  mPtCorrOverReco_Pt_E                = 0;
  mPtCorrOverReco_Pt_F                = 0;
  mPtCorrOverReco_Eta_20_40          = 0;
  mPtCorrOverReco_Eta_40_200          = 0;
  mPtCorrOverReco_Eta_200_600         = 0;
  mPtCorrOverReco_Eta_600_1500        = 0;
  mPtCorrOverReco_Eta_1500_3500       = 0;
  mPtCorrOverGen_GenPt_B          = 0;
  mPtCorrOverGen_GenPt_E          = 0;
  mPtCorrOverGen_GenPt_F          = 0;
  mPtCorrOverGen_GenEta_20_40    = 0;
  mPtCorrOverGen_GenEta_40_200    = 0;
  mPtCorrOverGen_GenEta_200_600   = 0;
  mPtCorrOverGen_GenEta_600_1500  = 0;
  mPtCorrOverGen_GenEta_1500_3500 = 0;

  // Generation
  mGenEta      = 0;
  mGenPhi      = 0;
  mGenPt       = 0;
  mGenEtaFirst = 0;
  mGenPhiFirst = 0;
  mPtHat       = 0;
  mDeltaEta    = 0;
  mDeltaPhi    = 0;
  mDeltaPt     = 0;

  mPtRecoOverGen_B_20_40    = 0;
  mPtRecoOverGen_E_20_40    = 0;
  mPtRecoOverGen_F_20_40    = 0;
  mPtRecoOverGen_B_40_200    = 0;
  mPtRecoOverGen_E_40_200    = 0;
  mPtRecoOverGen_F_40_200    = 0;
  mPtRecoOverGen_B_200_600   = 0;
  mPtRecoOverGen_E_200_600   = 0;
  mPtRecoOverGen_F_200_600   = 0;
  mPtRecoOverGen_B_600_1500  = 0;
  mPtRecoOverGen_E_600_1500  = 0;
  mPtRecoOverGen_F_600_1500  = 0;
  mPtRecoOverGen_B_1500_3500 = 0;
  mPtRecoOverGen_E_1500_3500 = 0;
  mPtRecoOverGen_F_1500_3500 = 0;

  // Generation profiles
  mPtRecoOverGen_GenPt_B          = 0;
  mPtRecoOverGen_GenPt_E          = 0;
  mPtRecoOverGen_GenPt_F          = 0;
  mPtRecoOverGen_GenPhi_B         = 0;
  mPtRecoOverGen_GenPhi_E         = 0;
  mPtRecoOverGen_GenPhi_F         = 0;
  mPtRecoOverGen_GenEta_20_40    = 0;
  mPtRecoOverGen_GenEta_40_200    = 0;
  mPtRecoOverGen_GenEta_200_600   = 0;
  mPtRecoOverGen_GenEta_600_1500  = 0;
  mPtRecoOverGen_GenEta_1500_3500 = 0;

  // Some jet algebra
  mEtaFirst   = 0;
  mPhiFirst   = 0;
  mPtFirst    = 0;
  mMjj        = 0;
  mNJetsEta_B_20_40 = 0;
  mNJetsEta_E_20_40 = 0;
  mNJetsEta_B_40 = 0;
  mNJetsEta_E_40 = 0;
  mNJets1     = 0;
  mNJets2     = 0;

//  // PFJet specific
//  mHadEnergyInHF       = 0;
//  mEmEnergyInHF        = 0;
//  mChargedEmEnergy     = 0;
//  mChargedHadronEnergy = 0;
//  mNeutralEmEnergy     = 0;
//  mNeutralHadronEnergy = 0;

  // ---- Calo Jet specific information ----
  /// returns the maximum energy deposited in ECAL towers
  maxEInEmTowers = 0;
  /// returns the maximum energy deposited in HCAL towers
  maxEInHadTowers = 0;
  /// returns the jet hadronic energy fraction
  energyFractionHadronic = 0;
  /// returns the jet electromagnetic energy fraction
  emEnergyFraction = 0;
  /// returns the jet hadronic energy in HB
  hadEnergyInHB = 0;
  /// returns the jet hadronic energy in HO
  hadEnergyInHO = 0;
  /// returns the jet hadronic energy in HE
  hadEnergyInHE = 0;
  /// returns the jet hadronic energy in HF
  hadEnergyInHF = 0;
  /// returns the jet electromagnetic energy in EB
  emEnergyInEB = 0;
  /// returns the jet electromagnetic energy in EE
  emEnergyInEE = 0;
  /// returns the jet electromagnetic energy extracted from HF
  emEnergyInHF = 0;
  /// returns area of contributing towers
  towersArea = 0;
  /// returns the number of constituents carrying a 90% of the total Jet energy*/
  n90 = 0;
  /// returns the number of constituents carrying a 60% of the total Jet energy*/
  n60 = 0;

  // ---- JPT Jet specific information ----
  /// chargedMultiplicity
  elecMultiplicity = 0;

  // ---- JPT or PF Jet specific information ----
  /// muonMultiplicity
  muonMultiplicity = 0;
  /// chargedMultiplicity
  chargedMultiplicity = 0;
  /// chargedEmEnergy
  chargedEmEnergy = 0;
  /// neutralEmEnergy
  neutralEmEnergy = 0;
  /// chargedHadronEnergy
  chargedHadronEnergy = 0;
  /// neutralHadronEnergy
  neutralHadronEnergy = 0;
  /// chargedHadronEnergyFraction (relative to uncorrected jet energy)
  chargedHadronEnergyFraction = 0;
  /// neutralHadronEnergyFraction (relative to uncorrected jet energy)
  neutralHadronEnergyFraction = 0;
  /// chargedEmEnergyFraction (relative to uncorrected jet energy)
  chargedEmEnergyFraction = 0;
  /// neutralEmEnergyFraction (relative to uncorrected jet energy)
  neutralEmEnergyFraction = 0;

  // ---- PF Jet specific information ----
  /// photonEnergy
  photonEnergy = 0;
  /// photonEnergyFraction (relative to corrected jet energy)
  photonEnergyFraction = 0;
  /// electronEnergy
  electronEnergy = 0;
  /// electronEnergyFraction (relative to corrected jet energy)
  electronEnergyFraction = 0;
  /// muonEnergy
  muonEnergy = 0;
  /// muonEnergyFraction (relative to corrected jet energy)
  muonEnergyFraction = 0;
  /// HFHadronEnergy
  HFHadronEnergy = 0;
  /// HFHadronEnergyFraction (relative to corrected jet energy)
  HFHadronEnergyFraction = 0;
  /// HFEMEnergy
  HFEMEnergy = 0;
  /// HFEMEnergyFraction (relative to corrected jet energy)
  HFEMEnergyFraction = 0;
  /// chargedHadronMultiplicity
  chargedHadronMultiplicity = 0;
  /// neutralHadronMultiplicity
  neutralHadronMultiplicity = 0;
  /// photonMultiplicity
  photonMultiplicity = 0;
  /// electronMultiplicity
  electronMultiplicity = 0;
  /// HFHadronMultiplicity
  HFHadronMultiplicity = 0;
  /// HFEMMultiplicity
  HFEMMultiplicity = 0;
  /// chargedMuEnergy
  chargedMuEnergy = 0;
  /// chargedMuEnergyFraction
  chargedMuEnergyFraction = 0;
  /// neutralMultiplicity
  neutralMultiplicity = 0;


  DQMStore* dbe = &*edm::Service<DQMStore>();
    
  if (dbe) {
    dbe->setCurrentFolder("JetMET/JetValidation/"+mInputCollection.label());

    double log10PtMin  = 0.50;
    double log10PtMax  = 3.75;
    int    log10PtBins = 26; 

    double etaRange[91] = {-6.0, -5.8, -5.6, -5.4, -5.2, -5.0, -4.8, -4.6, -4.4, -4.2,
		     -4.0, -3.8, -3.6, -3.4, -3.2, -3.0, -2.9, -2.8, -2.7, -2.6,
		     -2.5, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6,
		     -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6,
		     -0.5, -0.4, -0.3, -0.2, -0.1,
		     0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
		     1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
		     2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
		     3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8,
		     5.0, 5.2, 5.4, 5.6, 5.8, 6.0};

    // Event variables
    mNvtx           = dbe->book1D("Nvtx",           "number of vertices", 60, 0, 60);

    // Jet parameters
    mEta          = dbe->book1D("Eta",          "Eta",          120,   -6,    6); 
    mPhi          = dbe->book1D("Phi",          "Phi",           70, -3.5,  3.5); 
    mPt           = dbe->book1D("Pt",           "Pt",           100,    0,  1000); 
    mP            = dbe->book1D("P",            "P",            100,    0,  1000); 
    mEnergy       = dbe->book1D("Energy",       "Energy",       100,    0,  1000); 
    mMass         = dbe->book1D("Mass",         "Mass",         100,    0,  200); 
    mConstituents = dbe->book1D("Constituents", "Constituents", 100,    0,  100); 
    mHadTiming    = dbe->book1D("HadTiming",    "HadTiming",     75,  -50,  100);
    mEmTiming     = dbe->book1D("EmTiming",     "EmTiming",      75,  -50,  100);
    mJetArea      = dbe->book1D("JetArea",      "JetArea",       100,   0, 4);
//    mRho          = dbe->book1D("Rho",          "Rho",           100,    0,   5);

    // Corrected jets
    if (!JetCorrectionService.empty())	{
      mCorrJetPt  = dbe->book1D("CorrJetPt",  "CorrJetPt",  150,    0, 1500);
      mCorrJetEta = dbe->book1D("CorrJetEta", "CorrJetEta Pt>20", 60,   -6,   6);
      mCorrJetPhi = dbe->book1D("CorrJetPhi", "CorrJetPhi Pt>20",  70, -3.5, 3.5);
      mCorrJetEta_Pt40 = dbe->book1D("CorrJetEta_Pt40", "CorrJetEta Pt>40", 60,   -6,   6);
      mCorrJetPhi_Pt40 = dbe->book1D("CorrJetPhi_Pt40", "CorrJetPhi Pt>40",  70, -3.5, 3.5);

      // Corrected jets profiles
      mPtCorrOverReco_Pt_B = dbe->bookProfile("PtCorrOverReco_Pt_B", "0<|eta|<1.5", log10PtBins, log10PtMin, log10PtMax, 0, 5, " ");
      mPtCorrOverReco_Pt_E = dbe->bookProfile("PtCorrOverReco_Pt_E", "1.5<|eta|<3", log10PtBins, log10PtMin, log10PtMax, 0, 5, " ");
      mPtCorrOverReco_Pt_F = dbe->bookProfile("PtCorrOverReco_Pt_F", "3<|eta|<6",   log10PtBins, log10PtMin, log10PtMax, 0, 5, " ");

      mPtCorrOverReco_Eta_20_40    = dbe->bookProfile("PtCorrOverReco_Eta_20_40",    "20<genPt<40",    90, etaRange, 0, 5, " ");
      mPtCorrOverReco_Eta_40_200    = dbe->bookProfile("PtCorrOverReco_Eta_40_200",    "40<genPt<200",    90, etaRange, 0, 5, " ");
      mPtCorrOverReco_Eta_200_600   = dbe->bookProfile("PtCorrOverReco_Eta_200_600",   "200<genPt<600",   90, etaRange, 0, 5, " ");
      mPtCorrOverReco_Eta_600_1500  = dbe->bookProfile("PtCorrOverReco_Eta_600_1500",  "600<genPt<1500",  90, etaRange, 0, 5, " ");
      mPtCorrOverReco_Eta_1500_3500 = dbe->bookProfile("PtCorrOverReco_Eta_1500_3500", "1500<genPt<3500", 90, etaRange, 0, 5, " ");

      mPtCorrOverGen_GenPt_B = dbe->bookProfile("PtCorrOverGen_GenPt_B", "0<|eta|<1.5", log10PtBins, log10PtMin, log10PtMax, 0.8, 1.2, " ");
      mPtCorrOverGen_GenPt_E = dbe->bookProfile("PtCorrOverGen_GenPt_E", "1.5<|eta|<3", log10PtBins, log10PtMin, log10PtMax, 0.8, 1.2, " ");
      mPtCorrOverGen_GenPt_F = dbe->bookProfile("PtCorrOverGen_GenPt_F", "3<|eta|<6",   log10PtBins, log10PtMin, log10PtMax, 0.8, 1.2, " ");

      mPtCorrOverGen_GenEta_20_40    = dbe->bookProfile("PtCorrOverGen_GenEta_20_40",      "20<genPt<40;#eta",    90, etaRange, 0.8, 1.2, " ");
      mPtCorrOverGen_GenEta_40_200    = dbe->bookProfile("PtCorrOverGen_GenEta_40_200",    "40<genPt<200;#eta",    90, etaRange, 0.8, 1.2, " ");
      mPtCorrOverGen_GenEta_200_600   = dbe->bookProfile("PtCorrOverGen_GenEta_200_600",   "200<genPt<600;#eta",   90, etaRange, 0.8, 1.2, " ");
      mPtCorrOverGen_GenEta_600_1500  = dbe->bookProfile("PtCorrOverGen_GenEta_600_1500",  "600<genPt<1500;#eta",  90, etaRange, 0.8, 1.2, " ");
      mPtCorrOverGen_GenEta_1500_3500 = dbe->bookProfile("PtCorrOverGen_GenEta_1500_3500", "1500<genPt<3500;#eta", 90, etaRange, 0.8, 1.2, " ");
    }

    mGenEta      = dbe->book1D("GenEta",      "GenEta",      120,   -6,    6);
    mGenPhi      = dbe->book1D("GenPhi",      "GenPhi",       70, -3.5,  3.5);
    mGenPt       = dbe->book1D("GenPt",       "GenPt",       100,    0,  1000);
    mGenEtaFirst = dbe->book1D("GenEtaFirst", "GenEtaFirst", 120,   -6,    6);
    mGenPhiFirst = dbe->book1D("GenPhiFirst", "GenPhiFirst",  70, -3.5,  3.5);
    mPtHat       = dbe->book1D("PtHat",       "PtHat",       100,    0, 1000); 
    mDeltaEta    = dbe->book1D("DeltaEta",    "DeltaEta",    100, -0.5,  0.5);
    mDeltaPhi    = dbe->book1D("DeltaPhi",    "DeltaPhi",    100, -0.5,  0.5);
    mDeltaPt     = dbe->book1D("DeltaPt",     "DeltaPt",     100, -1.0,  1.0);
    
    mPtRecoOverGen_B_20_40    = dbe->book1D("PtRecoOverGen_B_20_40",    "20<genpt<40",    50, 0, 2);
    mPtRecoOverGen_E_20_40    = dbe->book1D("PtRecoOverGen_E_20_40",    "20<genpt<40",    50, 0, 2);
    mPtRecoOverGen_F_20_40    = dbe->book1D("PtRecoOverGen_F_20_40",    "20<genpt<40",    50, 0, 2);
    mPtRecoOverGen_B_40_200    = dbe->book1D("PtRecoOverGen_B_40_200",    "40<genpt<200",    50, 0, 2);
    mPtRecoOverGen_E_40_200    = dbe->book1D("PtRecoOverGen_E_40_200",    "40<genpt<200",    50, 0, 2);
    mPtRecoOverGen_F_40_200    = dbe->book1D("PtRecoOverGen_F_40_200",    "40<genpt<200",    50, 0, 2);
    mPtRecoOverGen_B_200_600   = dbe->book1D("PtRecoOverGen_B_200_600",   "200<genpt<600",   50, 0, 2);
    mPtRecoOverGen_E_200_600   = dbe->book1D("PtRecoOverGen_E_200_600",   "200<genpt<600",   50, 0, 2);
    mPtRecoOverGen_F_200_600   = dbe->book1D("PtRecoOverGen_F_200_600",   "200<genpt<600",   50, 0, 2);
    mPtRecoOverGen_B_600_1500  = dbe->book1D("PtRecoOverGen_B_600_1500",  "600<genpt<1500",  50, 0, 2);
    mPtRecoOverGen_E_600_1500  = dbe->book1D("PtRecoOverGen_E_600_1500",  "600<genpt<1500",  50, 0, 2);
    mPtRecoOverGen_F_600_1500  = dbe->book1D("PtRecoOverGen_F_600_1500",  "600<genpt<1500",  50, 0, 2);
    mPtRecoOverGen_B_1500_3500 = dbe->book1D("PtRecoOverGen_B_1500_3500", "1500<genpt<3500", 50, 0, 2);
    mPtRecoOverGen_E_1500_3500 = dbe->book1D("PtRecoOverGen_E_1500_3500", "1500<genpt<3500", 50, 0, 2);
    mPtRecoOverGen_F_1500_3500 = dbe->book1D("PtRecoOverGen_F_1500_3500", "1500<genpt<3500", 50, 0, 2);

    // Generation profiles
    mPtRecoOverGen_GenPt_B          = dbe->bookProfile("PtRecoOverGen_GenPt_B",          "0<|eta|<1.5",     log10PtBins, log10PtMin, log10PtMax, 0, 2, " ");
    mPtRecoOverGen_GenPt_E          = dbe->bookProfile("PtRecoOverGen_GenPt_E",          "1.5<|eta|<3",     log10PtBins, log10PtMin, log10PtMax, 0, 2, " ");
    mPtRecoOverGen_GenPt_F          = dbe->bookProfile("PtRecoOverGen_GenPt_F",          "3<|eta|<6",       log10PtBins, log10PtMin, log10PtMax, 0, 2, " ");
    mPtRecoOverGen_GenPhi_B         = dbe->bookProfile("PtRecoOverGen_GenPhi_B",         "0<|eta|<1.5",     70, -3.5, 3.5, 0, 2, " ");
    mPtRecoOverGen_GenPhi_E         = dbe->bookProfile("PtRecoOverGen_GenPhi_E",         "1.5<|eta|<3",     70, -3.5, 3.5, 0, 2, " ");
    mPtRecoOverGen_GenPhi_F         = dbe->bookProfile("PtRecoOverGen_GenPhi_F",         "3<|eta|<6",       70, -3.5, 3.5, 0, 2, " ");
    mPtRecoOverGen_GenEta_20_40    = dbe->bookProfile("PtRecoOverGen_GenEta_20_40",    "20<genpt<40",    50, etaRange, 0, 2, " ");
    mPtRecoOverGen_GenEta_40_200    = dbe->bookProfile("PtRecoOverGen_GenEta_40_200",    "40<genpt<200",    50, etaRange, 0, 2, " ");
    mPtRecoOverGen_GenEta_200_600   = dbe->bookProfile("PtRecoOverGen_GenEta_200_600",   "200<genpt<600",   50, etaRange, 0, 2, " ");
    mPtRecoOverGen_GenEta_600_1500  = dbe->bookProfile("PtRecoOverGen_GenEta_600_1500",  "600<genpt<1500",  50, etaRange, 0, 2, " ");
    mPtRecoOverGen_GenEta_1500_3500 = dbe->bookProfile("PtRecoOverGen_GenEta_1500_3500", "1500<genpt<3500", 50, etaRange, 0, 2, " ");
    
    // Some jet algebra
    //------------------------------------------------------------------------
    mEtaFirst   = dbe->book1D("EtaFirst",   "EtaFirst",   120,   -6,    6); 
    mPhiFirst   = dbe->book1D("PhiFirst",   "PhiFirst",    70, -3.5,  3.5);      
    mPtFirst    = dbe->book1D("PtFirst",    "PtFirst",    50,    0,  1000); 
    mMjj        = dbe->book1D("Mjj",        "Mjj",        100,    0, 2000); 
    mNJetsEta_B_20_40 = dbe->book1D("NJetsEta_B_20_40", "NJetsEta_B 20<Pt<40",  15,    0,   15);
    mNJetsEta_E_20_40 = dbe->book1D("NJetsEta_E_20_40", "NJetsEta_E 20<Pt<40",  15,    0,   15);
    mNJetsEta_B_40 = dbe->book1D("NJetsEta_B", "NJetsEta_B 40<Pt",  15,    0,   15);
    mNJetsEta_E_40 = dbe->book1D("NJetsEta_E", "NJetsEta_E 40<Pt",  15,    0,   15);
    mNJets_40 = dbe->book1D("NJets", "NJets 40>Pt",  15,    0,   15);
    mNJets1 = dbe->bookProfile("NJets1", "Number of jets above Pt threshold", 100, 0,  200, 100, 0, 50, "s");
    mNJets2 = dbe->bookProfile("NJets2", "Number of jets above Pt threshold", 100, 0, 4000, 100, 0, 50, "s");


    // PFJet specific
    //------------------------------------------------------------------------
//    if (isPFJet) {
//      mChargedEmEnergy     = dbe->book1D("ChargedEmEnergy",     "ChargedEmEnergy",     100,   0,  500);
//      mChargedHadronEnergy = dbe->book1D("ChargedHadronEnergy", "ChargedHadronEnergy", 100,   0,  500);
//      mNeutralEmEnergy     = dbe->book1D("NeutralEmEnergy",     "NeutralEmEnergy",     100,   0,  500);
//      mNeutralHadronEnergy = dbe->book1D("NeutralHadronEnergy", "NeutralHadronEnergy", 100,   0,  500);
//      mHadEnergyInHF       = dbe->book1D("HadEnergyInHF",       "HadEnergyInHF",       100,   0, 2500); 
//      mEmEnergyInHF        = dbe->book1D("EmEnergyInHF",        "EmEnergyInHF",        100, -20,  450); 
//    }
    // ---- Calo Jet specific information ----
    if (isCaloJet) {
      maxEInEmTowers              = dbe->book1D("maxEInEmTowers", "maxEInEmTowers", 50,0,500);
      maxEInHadTowers             = dbe->book1D("maxEInHadTowers", "maxEInHadTowers", 50,0,500);
      energyFractionHadronic      = dbe->book1D("energyFractionHadronic", "energyFractionHadronic", 50,0,1);
      emEnergyFraction            = dbe->book1D("emEnergyFraction", "emEnergyFraction", 50,0,1);
      hadEnergyInHB               = dbe->book1D("hadEnergyInHB", "hadEnergyInHB", 50,0,500);
      hadEnergyInHO               = dbe->book1D("hadEnergyInHO", "hadEnergyInHO", 50,0,500);
      hadEnergyInHE               = dbe->book1D("hadEnergyInHE", "hadEnergyInHE", 50,0,500);
      hadEnergyInHF               = dbe->book1D("hadEnergyInHF", "hadEnergyInHF", 50,0,500);
      emEnergyInEB                = dbe->book1D("emEnergyInEB", "emEnergyInEB", 50,0,500);
      emEnergyInEE                = dbe->book1D("emEnergyInEE", "emEnergyInEE", 50,0,500);
      emEnergyInHF                = dbe->book1D("emEnergyInHF", "emEnergyInHF", 50,0,500);
      towersArea                  = dbe->book1D("towersArea", "towersArea", 50,0,1);
      n90                         = dbe->book1D("n90", "n90", 30,0,30);
      n60                         = dbe->book1D("n60", "n60", 30,0,30);
    }
    // ---- JPT Jet specific information ----
    if (isJPTJet) {
      elecMultiplicity = dbe->book1D("elecMultiplicity", "elecMultiplicity", 10,0,10);
    }
    // ---- JPT or PF Jet specific information ----
    if (isPFJet or isJPTJet) {
      muonMultiplicity = dbe->book1D("muonMultiplicity", "muonMultiplicity", 10,0,10);
      chargedMultiplicity = dbe->book1D("chargedMultiplicity", "chargedMultiplicity", 100,0,100);
      chargedEmEnergy = dbe->book1D("chargedEmEnergy", "chargedEmEnergy", 100,0,500);
      neutralEmEnergy = dbe->book1D("neutralEmEnergy", "neutralEmEnergy", 100,0,500);
      chargedHadronEnergy = dbe->book1D("chargedHadronEnergy", "chargedHadronEnergy", 100,0,500);
      neutralHadronEnergy = dbe->book1D("neutralHadronEnergy", "neutralHadronEnergy", 100,0,500);
      chargedHadronEnergyFraction = dbe->book1D("chargedHadronEnergyFraction", "chargedHadronEnergyFraction", 50,0,1);
      neutralHadronEnergyFraction = dbe->book1D("neutralHadronEnergyFraction", "neutralHadronEnergyFraction", 50,0,1);
      chargedEmEnergyFraction = dbe->book1D("chargedEmEnergyFraction", "chargedEmEnergyFraction", 50,0,1);
      neutralEmEnergyFraction = dbe->book1D("neutralEmEnergyFraction", "neutralEmEnergyFraction", 50,0,1);
    }
    // ---- PF Jet specific information ----
    if (isPFJet) {
      photonEnergy = dbe->book1D("photonEnergy", "photonEnergy", 50,0,500);
      photonEnergyFraction = dbe->book1D("photonEnergyFraction", "photonEnergyFraction", 50,0,1);
      electronEnergy = dbe->book1D("electronEnergy", "electronEnergy", 50,0,500);
      electronEnergyFraction = dbe->book1D("electronEnergyFraction", "electronEnergyFraction", 50,0,1);
      muonEnergy = dbe->book1D("muonEnergy", "muonEnergy", 50,0,500);
      muonEnergyFraction = dbe->book1D("muonEnergyFraction", "muonEnergyFraction", 50,0,1);
      HFHadronEnergy = dbe->book1D("HFHadronEnergy", "HFHadronEnergy", 50,0,500);
      HFHadronEnergyFraction = dbe->book1D("HFHadronEnergyFraction", "HFHadronEnergyFraction", 50,0,1);
      HFEMEnergy = dbe->book1D("HFEMEnergy", "HFEMEnergy", 50,0,500);
      HFEMEnergyFraction = dbe->book1D("HFEMEnergyFraction", "HFEMEnergyFraction", 50,0,1);
      chargedHadronMultiplicity = dbe->book1D("chargedHadronMultiplicity", "chargedHadronMultiplicity", 50,0,50);
      neutralHadronMultiplicity = dbe->book1D("neutralHadronMultiplicity", "neutralHadronMultiplicity", 50,0,50);
      photonMultiplicity = dbe->book1D("photonMultiplicity", "photonMultiplicity", 10,0,10);
      electronMultiplicity = dbe->book1D("electronMultiplicity", "electronMultiplicity", 10,0,10);
      HFHadronMultiplicity = dbe->book1D("HFHadronMultiplicity", "HFHadronMultiplicity", 50,0,50);
      HFEMMultiplicity = dbe->book1D("HFEMMultiplicity", "HFEMMultiplicity", 50,0,50);
      chargedMuEnergy = dbe->book1D("chargedMuEnergy", "chargedMuEnergy", 50,0,500);
      chargedMuEnergyFraction = dbe->book1D("chargedMuEnergyFraction", "chargedMuEnergyFraction", 50,0,1);
      neutralMultiplicity = dbe->book1D("neutralMultiplicity", "neutralMultiplicity", 50,0,50);
    }
  }

  if (mOutputFile.empty ())
    {
      LogInfo("OutputInfo") << " Histograms will NOT be saved";
    }
  else
    {
      LogInfo("OutputInfo") << " Histograms will be saved to file:" << mOutputFile;
    }
}


//------------------------------------------------------------------------------
// ~JetTester
//------------------------------------------------------------------------------
JetTester::~JetTester() {}


//------------------------------------------------------------------------------
// beginJob
//------------------------------------------------------------------------------
void JetTester::beginJob() {}


//------------------------------------------------------------------------------
// endJob
//------------------------------------------------------------------------------
void JetTester::endJob()
{
  if (!mOutputFile.empty() && &*edm::Service<DQMStore>())
    {
      edm::Service<DQMStore>()->save(mOutputFile);
    }
}


//------------------------------------------------------------------------------
// analyze
//------------------------------------------------------------------------------
void JetTester::analyze(const edm::Event& mEvent, const edm::EventSetup& mSetup)
{
  // Get the primary vertices
  //----------------------------------------------------------------------------
  edm::Handle<vector<reco::Vertex> > pvHandle;
  mEvent.getByToken(pvToken_, pvHandle);

  int nGoodVertices = 0;

  if (pvHandle.isValid())
    {
      for (unsigned i=0; i<pvHandle->size(); i++)
	{
	  if ((*pvHandle)[i].ndof() > 4 &&
	      (fabs((*pvHandle)[i].z()) <= 24) &&
	      (fabs((*pvHandle)[i].position().rho()) <= 2))
	    nGoodVertices++;
	}
    }

  mNvtx->Fill(nGoodVertices);


  // Get the CaloTower collection
  //----------------------------------------------------------------------------
  edm::Handle<CaloTowerCollection> caloTowers;
  mEvent.getByToken(caloTowersToken_, caloTowers);

  if (caloTowers.isValid())
    {
      for (CaloTowerCollection::const_iterator cal=caloTowers->begin();
	   cal!=caloTowers->end(); ++cal)
	{
	  mHadTiming->Fill(cal->hcalTime());
	  mEmTiming ->Fill(cal->ecalTime());    
	}
    }  


//  // Get the jet rho
//  //----------------------------------------------------------------------------
//  edm::Handle<double> pRho;
//  mEvent.getByToken(rhoTag, pRho);
//
//  if (pRho.isValid())
//    {
//      double jetRho = *pRho;
//
//      if (mRho) mRho->Fill(jetRho);
//    }


  // Get the Jet collection
  //----------------------------------------------------------------------------
  math::XYZTLorentzVector p4tmp[2];

  std::vector<Jet> recoJets;
  recoJets.clear();

  edm::Handle<CaloJetCollection> caloJets;
  edm::Handle<JPTJetCollection>  jptJets;
  edm::Handle<PFJetCollection>   pfJets;

  if (isCaloJet) mEvent.getByToken(caloJetsToken_, caloJets);
  if (isJPTJet)  mEvent.getByToken(jptJetsToken_, jptJets);
  if (isPFJet)   mEvent.getByToken(pfJetsToken_, pfJets);

  if (isCaloJet && !caloJets.isValid()) return;
  if (isJPTJet  && !jptJets.isValid())  return;
  if (isPFJet   && !pfJets.isValid())   return;

  if (isCaloJet)
    {
      for (unsigned ijet=0; ijet<caloJets->size(); ijet++)
	recoJets.push_back((*caloJets)[ijet]);
    }

  if (isJPTJet)
    {
      for (unsigned ijet=0; ijet<jptJets->size(); ijet++)
	recoJets.push_back((*jptJets)[ijet]);
    }

  if (isPFJet) {
    for (unsigned ijet=0; ijet<pfJets->size(); ijet++)
      recoJets.push_back((*pfJets)[ijet]);
  }

  int nJet      = 0;
  int nJet_E_20_40 = 0;
  int nJet_B_20_40 = 0;
  int nJet_E_40 = 0;
  int nJet_B_40 = 0;
  int nJet_40 = 0;

  for (unsigned ijet=0; ijet<recoJets.size(); ijet++) {
    if (  (recoJets[ijet].pt() > 20.) and  (recoJets[ijet].pt() < mRecoJetPtThreshold)) {
      if (fabs(recoJets[ijet].eta()) > 1.5)
        nJet_E_20_40++;
      else
        nJet_B_20_40++;	  
    }
    if (recoJets[ijet].pt() > mRecoJetPtThreshold) {
      //counting forward and barrel jets
      if (fabs(recoJets[ijet].eta()) > 1.5)
        nJet_E_40++;
      else
        nJet_B_40++;	  
      nJet_40++;

      if (mEta) mEta->Fill(recoJets[ijet].eta());

      if (mJetArea)      mJetArea     ->Fill(recoJets[ijet].jetArea());
      if (mPhi)          mPhi         ->Fill(recoJets[ijet].phi());
      if (mEnergy)       mEnergy      ->Fill(recoJets[ijet].energy());
      if (mP)            mP           ->Fill(recoJets[ijet].p());
      if (mPt)           mPt          ->Fill(recoJets[ijet].pt());
      if (mMass)         mMass        ->Fill(recoJets[ijet].mass());
      if (mConstituents) mConstituents->Fill(recoJets[ijet].nConstituents());

      if (ijet == 0) {
        if (mEtaFirst) mEtaFirst->Fill(recoJets[ijet].eta());
        if (mPhiFirst) mPhiFirst->Fill(recoJets[ijet].phi());
        if (mPtFirst)  mPtFirst ->Fill(recoJets[ijet].pt());
      }

      if (ijet == 0) {nJet++; p4tmp[0] = recoJets[ijet].p4();}
      if (ijet == 1) {nJet++; p4tmp[1] = recoJets[ijet].p4();}
      
  //    if (isPFJet || isCaloJet) {
  //      if (mHadEnergyInHF)       mHadEnergyInHF      ->Fill((*pfJets)[ijet].HFHadronEnergy());
  //      if (mEmEnergyInHF)        mEmEnergyInHF       ->Fill((*pfJets)[ijet].HFEMEnergy());
  //      if (mChargedEmEnergy)     mChargedEmEnergy    ->Fill((*pfJets)[ijet].chargedEmEnergy());
  //      if (mChargedHadronEnergy) mChargedHadronEnergy->Fill((*pfJets)[ijet].chargedHadronEnergy());
  //      if (mNeutralEmEnergy)     mNeutralEmEnergy    ->Fill((*pfJets)[ijet].neutralEmEnergy());
  //      if (mNeutralHadronEnergy) mNeutralHadronEnergy->Fill((*pfJets)[ijet].neutralHadronEnergy());
  //    }


      // ---- Calo Jet specific information ----
      if (isCaloJet) {
        maxEInEmTowers              ->Fill((*caloJets)[ijet].maxEInEmTowers());
        maxEInHadTowers             ->Fill((*caloJets)[ijet].maxEInHadTowers());
        energyFractionHadronic      ->Fill((*caloJets)[ijet].energyFractionHadronic());
        emEnergyFraction            ->Fill((*caloJets)[ijet].emEnergyFraction());
        hadEnergyInHB               ->Fill((*caloJets)[ijet].hadEnergyInHB());
        hadEnergyInHO               ->Fill((*caloJets)[ijet].hadEnergyInHO());
        hadEnergyInHE               ->Fill((*caloJets)[ijet].hadEnergyInHE());
        hadEnergyInHF               ->Fill((*caloJets)[ijet].hadEnergyInHF());
        emEnergyInEB                ->Fill((*caloJets)[ijet].emEnergyInEB());
        emEnergyInEE                ->Fill((*caloJets)[ijet].emEnergyInEE());
        emEnergyInHF                ->Fill((*caloJets)[ijet].emEnergyInHF());
        towersArea                  ->Fill((*caloJets)[ijet].towersArea());
        n90                         ->Fill((*caloJets)[ijet].n90());
        n60                         ->Fill((*caloJets)[ijet].n60());
      }
      // ---- JPT Jet specific information ----
      if (isJPTJet) {
        elecMultiplicity ->Fill((*jptJets)[ijet].elecMultiplicity());
      }
      // ---- JPT or PF Jet specific information ----
      if (isPFJet) {
        muonMultiplicity ->Fill((*pfJets)[ijet].muonMultiplicity());
        chargedMultiplicity ->Fill((*pfJets)[ijet].chargedMultiplicity());
        chargedEmEnergy ->Fill((*pfJets)[ijet].chargedEmEnergy());
        neutralEmEnergy ->Fill((*pfJets)[ijet].neutralEmEnergy());
        chargedHadronEnergy ->Fill((*pfJets)[ijet].chargedHadronEnergy());
        neutralHadronEnergy ->Fill((*pfJets)[ijet].neutralHadronEnergy());
        chargedHadronEnergyFraction ->Fill((*pfJets)[ijet].chargedHadronEnergyFraction());
        neutralHadronEnergyFraction ->Fill((*pfJets)[ijet].neutralHadronEnergyFraction());
        chargedEmEnergyFraction ->Fill((*pfJets)[ijet].chargedEmEnergyFraction());
        neutralEmEnergyFraction ->Fill((*pfJets)[ijet].neutralEmEnergyFraction());
      }
      if (isJPTJet) {
        muonMultiplicity ->Fill((*jptJets)[ijet].muonMultiplicity());
        chargedMultiplicity ->Fill((*jptJets)[ijet].chargedMultiplicity());
        chargedEmEnergy ->Fill((*jptJets)[ijet].chargedEmEnergy());
        neutralEmEnergy ->Fill((*jptJets)[ijet].neutralEmEnergy());
        chargedHadronEnergy ->Fill((*jptJets)[ijet].chargedHadronEnergy());
        neutralHadronEnergy ->Fill((*jptJets)[ijet].neutralHadronEnergy());
        chargedHadronEnergyFraction ->Fill((*jptJets)[ijet].chargedHadronEnergyFraction());
        neutralHadronEnergyFraction ->Fill((*jptJets)[ijet].neutralHadronEnergyFraction());
        chargedEmEnergyFraction ->Fill((*jptJets)[ijet].chargedEmEnergyFraction());
        neutralEmEnergyFraction ->Fill((*jptJets)[ijet].neutralEmEnergyFraction());
      }
      // ---- PF Jet specific information ----
      if (isPFJet) {
        photonEnergy ->Fill((*pfJets)[ijet].photonEnergy());
        photonEnergyFraction ->Fill((*pfJets)[ijet].photonEnergyFraction());
        electronEnergy ->Fill((*pfJets)[ijet].electronEnergy());
        electronEnergyFraction ->Fill((*pfJets)[ijet].electronEnergyFraction());
        muonEnergy ->Fill((*pfJets)[ijet].muonEnergy());
        muonEnergyFraction ->Fill((*pfJets)[ijet].muonEnergyFraction());
        HFHadronEnergy ->Fill((*pfJets)[ijet].HFHadronEnergy());
        HFHadronEnergyFraction ->Fill((*pfJets)[ijet].HFHadronEnergyFraction());
        HFEMEnergy ->Fill((*pfJets)[ijet].HFEMEnergy());
        HFEMEnergyFraction ->Fill((*pfJets)[ijet].HFEMEnergyFraction());
        chargedHadronMultiplicity ->Fill((*pfJets)[ijet].chargedHadronMultiplicity());
        neutralHadronMultiplicity ->Fill((*pfJets)[ijet].neutralHadronMultiplicity());
        photonMultiplicity ->Fill((*pfJets)[ijet].photonMultiplicity());
        electronMultiplicity ->Fill((*pfJets)[ijet].electronMultiplicity());
        HFHadronMultiplicity ->Fill((*pfJets)[ijet].HFHadronMultiplicity());
        HFEMMultiplicity ->Fill((*pfJets)[ijet].HFEMMultiplicity());
        chargedMuEnergy ->Fill((*pfJets)[ijet].chargedMuEnergy());
        chargedMuEnergyFraction ->Fill((*pfJets)[ijet].chargedMuEnergyFraction());
        neutralMultiplicity ->Fill((*pfJets)[ijet].neutralMultiplicity());
      }
    }
  }

  if (mNJetsEta_B_20_40) mNJetsEta_B_20_40->Fill(nJet_B_20_40);
  if (mNJetsEta_E_20_40) mNJetsEta_E_20_40->Fill(nJet_E_20_40);
  if (mNJetsEta_B_40) mNJetsEta_B_40->Fill(nJet_B_40);
  if (mNJetsEta_E_40) mNJetsEta_E_40->Fill(nJet_E_40);
  if (mNJets_40) mNJets_40->Fill(nJet_40); 
  if (nJet >= 2)
    {
      if (mMjj) mMjj->Fill((p4tmp[0]+p4tmp[1]).mass());
    }


  // Count jets above pt cut
  //----------------------------------------------------------------------------
  for (int istep=0; istep<100; ++istep)
    {
      int njets1 = 0;
      int njets2 = 0;

      float ptStep1 = (istep * ( 200. / 100.));
      float ptStep2 = (istep * (4000. / 100.));

      for (unsigned ijet=0; ijet<recoJets.size(); ijet++) {
        if (recoJets[ijet].pt() > ptStep1) njets1++;
        if (recoJets[ijet].pt() > ptStep2) njets2++;
      }
      mNJets1->Fill(ptStep1, njets1);
      mNJets2->Fill(ptStep2, njets2);
    }


  // Corrected jets
  //----------------------------------------------------------------------------
  double scale = -999;

  if (!JetCorrectionService.empty())
    {
      const JetCorrector* corrector = JetCorrector::getJetCorrector(JetCorrectionService, mSetup);
      for (unsigned ijet=0; ijet<recoJets.size(); ijet++) {
        Jet correctedJet = recoJets[ijet];

        if (isCaloJet) scale = corrector->correction((*caloJets)[ijet], mEvent, mSetup); 
        if (isJPTJet)  scale = corrector->correction((*jptJets)[ijet],  mEvent, mSetup); 
        if (isPFJet)   scale = corrector->correction((*pfJets)[ijet],   mEvent, mSetup); 

        correctedJet.scaleEnergy(scale); 
        
        if (correctedJet.pt() < 20) continue;

        mCorrJetEta->Fill(correctedJet.eta());
        mCorrJetPhi->Fill(correctedJet.phi());
        mCorrJetPt ->Fill(correctedJet.pt());
        if (correctedJet.pt() >= 40) {
          mCorrJetEta_Pt40->Fill(correctedJet.eta());
          mCorrJetPhi_Pt40->Fill(correctedJet.phi());
        }

        double ijetEta = recoJets[ijet].eta();
        double ijetPt  = recoJets[ijet].pt();
        double ratio   = correctedJet.pt() / ijetPt;

        if      (fabs(ijetEta) < 1.5) mPtCorrOverReco_Pt_B->Fill(log10(ijetPt), ratio);
        else if (fabs(ijetEta) < 3.0) mPtCorrOverReco_Pt_E->Fill(log10(ijetPt), ratio);
        else if (fabs(ijetEta) < 6.0) mPtCorrOverReco_Pt_F->Fill(log10(ijetPt), ratio);

        if      (ijetPt <  40) mPtCorrOverReco_Eta_20_40   ->Fill(ijetEta, ratio);
        else if (ijetPt <  200) mPtCorrOverReco_Eta_40_200  ->Fill(ijetEta, ratio);
        else if (ijetPt <  600) mPtCorrOverReco_Eta_200_600  ->Fill(ijetEta, ratio);
        else if (ijetPt < 1500) mPtCorrOverReco_Eta_600_1500 ->Fill(ijetEta, ratio);
        else if (ijetPt < 3500) mPtCorrOverReco_Eta_1500_3500->Fill(ijetEta, ratio);
      }
    }


  //----------------------------------------------------------------------------
  //
  // Generation
  //
  //----------------------------------------------------------------------------
  if (!mEvent.isRealData())
    {
      // Get ptHat
      //------------------------------------------------------------------------
      edm::Handle<HepMCProduct> evt;
      mEvent.getByToken(evtToken_, evt);

      if (evt.isValid()) {
        HepMC::GenEvent* myGenEvent = new HepMC::GenEvent(*(evt->GetEvent()));
        double ptHat = myGenEvent->event_scale();
        if (mPtHat) mPtHat->Fill(ptHat);
        delete myGenEvent; 
      }
      // Gen jets
      //------------------------------------------------------------------------
      edm::Handle<GenJetCollection> genJets;
      mEvent.getByToken(genJetsToken_, genJets);

      if (!genJets.isValid()) return;
      
      for (GenJetCollection::const_iterator gjet=genJets->begin();  gjet!=genJets->end(); gjet++)	{
        if (mGenEta) mGenEta->Fill(gjet->eta());
        if (mGenPhi) mGenPhi->Fill(gjet->phi());
        if (mGenPt)  mGenPt ->Fill(gjet->pt());
        if (gjet == genJets->begin()) {
          if (mGenEtaFirst) mGenEtaFirst->Fill(gjet->eta());
          if (mGenPhiFirst) mGenPhiFirst->Fill(gjet->phi());
        }
      }

      if (!(mInputGenCollection.label().empty())) {
      for (GenJetCollection::const_iterator gjet=genJets->begin(); gjet!=genJets->end(); gjet++) {
        if (fabs(gjet->eta()) > 6.) continue;  // Out of the detector 
        if (gjet->pt() < mMatchGenPtThreshold) continue;
        if (recoJets.size() <= 0) continue;
        // pt response
        //------------------------------------------------------------
        if (!JetCorrectionService.empty()) {
          int iMatch    =   -1;
          double CorrdeltaRBest = 999;
          double CorrJetPtBest  =   0;
          for (unsigned ijet=0; ijet<recoJets.size(); ++ijet) {
            Jet correctedJet = recoJets[ijet];
            correctedJet.scaleEnergy(scale);
            double CorrJetPt = correctedJet.pt();
            if (CorrJetPt > 10) {
              double CorrdR = deltaR(gjet->eta(), gjet->phi(), correctedJet.eta(), correctedJet.phi());
              if (CorrdR < CorrdeltaRBest) {
                CorrdeltaRBest = CorrdR;
                CorrJetPtBest  = CorrJetPt;
                iMatch = ijet;
              }
            }
          }
          if (iMatch<0) continue;
          fillMatchHists(gjet->eta(),  gjet->phi(),  gjet->pt(), recoJets[iMatch].eta(), recoJets[iMatch].phi(),  recoJets[iMatch].pt());
            
          if (CorrdeltaRBest < mRThreshold) {
            double response = CorrJetPtBest / gjet->pt();
            
            if      (fabs(gjet->eta()) < 1.5) mPtCorrOverGen_GenPt_B->Fill(log10(gjet->pt()), response);
            else if (fabs(gjet->eta()) < 3.0) mPtCorrOverGen_GenPt_E->Fill(log10(gjet->pt()), response);   
            else if (fabs(gjet->eta()) < 6.0) mPtCorrOverGen_GenPt_F->Fill(log10(gjet->pt()), response);
            
            if (gjet->pt() > 20) {
              if      (gjet->pt() <  40) mPtCorrOverGen_GenEta_20_40   ->Fill(gjet->eta(), response);
              else if (gjet->pt() <  200) mPtCorrOverGen_GenEta_40_200   ->Fill(gjet->eta(), response);
              else if (gjet->pt() <  600) mPtCorrOverGen_GenEta_200_600  ->Fill(gjet->eta(), response);
              else if (gjet->pt() < 1500) mPtCorrOverGen_GenEta_600_1500 ->Fill(gjet->eta(), response);
              else if (gjet->pt() < 3500) mPtCorrOverGen_GenEta_1500_3500->Fill(gjet->eta(), response);
            }
          }
        }
      }
    }
	}
}


//------------------------------------------------------------------------------
// fillMatchHists
//------------------------------------------------------------------------------
void JetTester::fillMatchHists(const double GenEta,
			       const double GenPhi,
			       const double GenPt,
			       const double RecoEta,
			       const double RecoPhi,
			       const double RecoPt) 
{
  if (GenPt > mMatchGenPtThreshold) {
    mDeltaEta->Fill(GenEta - RecoEta);
    mDeltaPhi->Fill(GenPhi - RecoPhi);
    mDeltaPt ->Fill((GenPt - RecoPt) / GenPt);
  }

  if (fabs(GenEta) < 1.5)
    {
      mPtRecoOverGen_GenPt_B ->Fill(log10(GenPt),  RecoPt / GenPt);
      mPtRecoOverGen_GenPhi_B->Fill(GenPhi, RecoPt / GenPt);
    
      if (GenPt > 20 && GenPt < 40) mPtRecoOverGen_B_20_40   ->Fill(RecoPt / GenPt);
      else if (GenPt <  200)         mPtRecoOverGen_B_40_200  ->Fill(RecoPt / GenPt);
      else if (GenPt <  600)         mPtRecoOverGen_B_200_600  ->Fill(RecoPt / GenPt);
      else if (GenPt < 1500)         mPtRecoOverGen_B_600_1500 ->Fill(RecoPt / GenPt);
      else if (GenPt < 3500)         mPtRecoOverGen_B_1500_3500->Fill(RecoPt / GenPt);
    }
  else if (fabs(GenEta) < 3.0)
    {
      mPtRecoOverGen_GenPt_E ->Fill(log10(GenPt),  RecoPt / GenPt);
      mPtRecoOverGen_GenPhi_E->Fill(GenPhi, RecoPt / GenPt);
    
      if (GenPt > 20 && GenPt < 40) mPtRecoOverGen_E_20_40   ->Fill(RecoPt / GenPt);
      else if (GenPt <  200)         mPtRecoOverGen_E_40_200  ->Fill(RecoPt / GenPt);
      else if (GenPt <  600)         mPtRecoOverGen_E_200_600  ->Fill(RecoPt / GenPt);
      else if (GenPt < 1500)         mPtRecoOverGen_E_600_1500 ->Fill(RecoPt / GenPt);
      else if (GenPt < 3500)         mPtRecoOverGen_E_1500_3500->Fill(RecoPt / GenPt);
    }
  else if (fabs(GenEta) < 6.0)
    {
      mPtRecoOverGen_GenPt_F ->Fill (log10(GenPt),  RecoPt / GenPt);
      mPtRecoOverGen_GenPhi_F->Fill (GenPhi, RecoPt / GenPt);
    
      if (GenPt > 20 && GenPt < 40) mPtRecoOverGen_F_20_40   ->Fill(RecoPt / GenPt);
      else if (GenPt <  200)         mPtRecoOverGen_F_40_200  ->Fill(RecoPt / GenPt);
      else if (GenPt <  600)         mPtRecoOverGen_F_200_600  ->Fill(RecoPt / GenPt);
      else if (GenPt < 1500)         mPtRecoOverGen_F_600_1500 ->Fill(RecoPt / GenPt);
      else if (GenPt < 3500)         mPtRecoOverGen_F_1500_3500->Fill(RecoPt / GenPt);
    }

  if (GenPt > 20 && GenPt < 40) mPtRecoOverGen_GenEta_20_40   ->Fill(GenEta, RecoPt / GenPt);
  else if (GenPt <  200)         mPtRecoOverGen_GenEta_40_200  ->Fill(GenEta, RecoPt / GenPt);
  else if (GenPt <  600)         mPtRecoOverGen_GenEta_200_600  ->Fill(GenEta, RecoPt / GenPt);
  else if (GenPt < 1500)         mPtRecoOverGen_GenEta_600_1500 ->Fill(GenEta, RecoPt / GenPt);
  else if (GenPt < 3500)         mPtRecoOverGen_GenEta_1500_3500->Fill(GenEta, RecoPt / GenPt);
}
