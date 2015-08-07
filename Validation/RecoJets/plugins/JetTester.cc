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
//  rhoTag                         (iConfig.getParameter<edm::InputTag>       ("srcRho")), 
  JetType                        (iConfig.getUntrackedParameter<std::string>("JetType")),
  mRecoJetPtThreshold            (iConfig.getParameter<double>              ("recoJetPtThreshold")),
  mMatchGenPtThreshold           (iConfig.getParameter<double>              ("matchGenPtThreshold")),
  mRThreshold                    (iConfig.getParameter<double>              ("RThreshold"))
{
  std::string inputCollectionLabel(mInputCollection.label());

  isCaloJet = (std::string("calo")==JetType);
  isPFJet   = (std::string("pf")  ==JetType);
  isMiniAODJet = (std::string("miniaod")  ==JetType);
  if(!isMiniAODJet){
    mJetCorrector                  =iConfig.getParameter<edm::InputTag>("JetCorrections"); 
  } 

  //consumes
  pvToken_ = consumes<std::vector<reco::Vertex> >(iConfig.getParameter<edm::InputTag>("primVertex"));
  if (isCaloJet) caloJetsToken_  = consumes<reco::CaloJetCollection>(mInputCollection);
  if (isPFJet)   pfJetsToken_    = consumes<reco::PFJetCollection>(mInputCollection);
  if(isMiniAODJet)patJetsToken_ = consumes<pat::JetCollection>(mInputCollection);
  mInputGenCollection            =iConfig.getParameter<edm::InputTag>("srcGen");
  genJetsToken_ = consumes<reco::GenJetCollection>(edm::InputTag(mInputGenCollection));
  evtToken_ = consumes<GenEventInfoProduct>(edm::InputTag("generator"));
  if(!isMiniAODJet && !mJetCorrector.label().empty()){
    jetCorrectorToken_ = consumes<reco::JetCorrector>(mJetCorrector);
  }
  
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
  mPtCorrOverReco_Eta_3500_5000       = 0;
  mPtCorrOverReco_Eta_5000_6500       = 0;
  mPtCorrOverReco_Eta_3500       = 0;
  mPtCorrOverGen_GenPt_B          = 0;
  mPtCorrOverGen_GenPt_E          = 0;
  mPtCorrOverGen_GenPt_F          = 0;
  mPtCorrOverGen_GenEta_20_40    = 0;
  mPtCorrOverGen_GenEta_40_200    = 0;
  mPtCorrOverGen_GenEta_200_600   = 0;
  mPtCorrOverGen_GenEta_600_1500  = 0;
  mPtCorrOverGen_GenEta_1500_3500 = 0;
  mPtCorrOverGen_GenEta_3500_5000 = 0;
  mPtCorrOverGen_GenEta_5000_6500 = 0;
  mPtCorrOverGen_GenEta_3500 = 0;

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
  mPtRecoOverGen_B_3500_5000 = 0;
  mPtRecoOverGen_E_3500_5000 = 0;
  mPtRecoOverGen_B_5000_6500 = 0;
  mPtRecoOverGen_E_5000_6500 = 0;
  mPtRecoOverGen_B_3500 = 0;
  mPtRecoOverGen_E_3500 = 0;
  mPtRecoOverGen_F_3500 = 0;

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
  mPtRecoOverGen_GenEta_3500_5000 = 0;
  mPtRecoOverGen_GenEta_5000_6500 = 0;
  mPtRecoOverGen_GenEta_3500 = 0;

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
//  elecMultiplicity = 0;

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

  /// HOEnergy
  HOEnergy = 0;
  /// HOEnergyFraction (relative to corrected jet energy)
  HOEnergyFraction = 0;
}

void JetTester::bookHistograms(DQMStore::IBooker & ibooker,
                                  edm::Run const & iRun,
                                  edm::EventSetup const & ) { 

  ibooker.setCurrentFolder("JetMET/JetValidation/"+mInputCollection.label());  

    double log10PtMin  = 0.50;
    double log10PtMax  = 3.75;
    int    log10PtBins = 26; 

    //if eta range changed here need change in JetTesterPostProcessor as well
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
    mNvtx           = ibooker.book1D("Nvtx",           "number of vertices", 60, 0, 60);

    // Jet parameters
    mEta          = ibooker.book1D("Eta",          "Eta",          120,   -6,    6); 
    mPhi          = ibooker.book1D("Phi",          "Phi",           70, -3.5,  3.5); 
    mPt           = ibooker.book1D("Pt",           "Pt",           100,    0,  1000); 
    mP            = ibooker.book1D("P",            "P",            100,    0,  1000); 
    mEnergy       = ibooker.book1D("Energy",       "Energy",       100,    0,  1000); 
    mMass         = ibooker.book1D("Mass",         "Mass",         100,    0,  200); 
    mConstituents = ibooker.book1D("Constituents", "Constituents", 100,    0,  100); 
    mJetArea      = ibooker.book1D("JetArea",      "JetArea",       100,   0, 4);

    // Corrected jets
    if (isMiniAODJet || !mJetCorrector.label().empty())	{//if correction label is filled, but fill also for MiniAOD though
      mCorrJetPt  = ibooker.book1D("CorrJetPt",  "CorrJetPt",  150,    0, 1500);
      mCorrJetEta = ibooker.book1D("CorrJetEta", "CorrJetEta Pt>20", 60,   -6,   6);
      mCorrJetPhi = ibooker.book1D("CorrJetPhi", "CorrJetPhi Pt>20",  70, -3.5, 3.5);
      mCorrJetEta_Pt40 = ibooker.book1D("CorrJetEta_Pt40", "CorrJetEta Pt>40", 60,   -6,   6);
      mCorrJetPhi_Pt40 = ibooker.book1D("CorrJetPhi_Pt40", "CorrJetPhi Pt>40",  70, -3.5, 3.5);

      // Corrected jets profiles
      mPtCorrOverReco_Pt_B = ibooker.bookProfile("PtCorrOverReco_Pt_B", "0<|eta|<1.5", log10PtBins, log10PtMin, log10PtMax, 0, 5, " ");
      mPtCorrOverReco_Pt_E = ibooker.bookProfile("PtCorrOverReco_Pt_E", "1.5<|eta|<3", log10PtBins, log10PtMin, log10PtMax, 0, 5, " ");
      mPtCorrOverReco_Pt_F = ibooker.bookProfile("PtCorrOverReco_Pt_F", "3<|eta|<6",   log10PtBins, log10PtMin, log10PtMax, 0, 5, " ");

      mPtCorrOverReco_Eta_20_40    = ibooker.bookProfile("PtCorrOverReco_Eta_20_40",    "20<genPt<40",    90, etaRange, 0, 5, " ");
      mPtCorrOverReco_Eta_40_200    = ibooker.bookProfile("PtCorrOverReco_Eta_40_200",    "40<genPt<200",    90, etaRange, 0, 5, " ");
      mPtCorrOverReco_Eta_200_600   = ibooker.bookProfile("PtCorrOverReco_Eta_200_600",   "200<genPt<600",   90, etaRange, 0, 5, " ");
      mPtCorrOverReco_Eta_600_1500  = ibooker.bookProfile("PtCorrOverReco_Eta_600_1500",  "600<genPt<1500",  90, etaRange, 0, 5, " ");
      mPtCorrOverReco_Eta_1500_3500 = ibooker.bookProfile("PtCorrOverReco_Eta_1500_3500", "1500<genPt<3500", 90, etaRange, 0, 5, " ");
      mPtCorrOverReco_Eta_3500_5000 = ibooker.bookProfile("PtCorrOverReco_Eta_3500_5000", "3500<genPt<5000", 90, etaRange, 0, 5, " ");
      mPtCorrOverReco_Eta_5000_6500 = ibooker.bookProfile("PtCorrOverReco_Eta_5000_6500", "5000<genPt<6500", 90, etaRange, 0, 5, " ");
      mPtCorrOverReco_Eta_3500      = ibooker.bookProfile("PtCorrOverReco_Eta_3500",      "genPt>3500",      90, etaRange, 0, 5, " ");

      mPtCorrOverGen_GenPt_B = ibooker.bookProfile("PtCorrOverGen_GenPt_B", "0<|eta|<1.5", log10PtBins, log10PtMin, log10PtMax, 0.8, 1.2, " ");
      mPtCorrOverGen_GenPt_E = ibooker.bookProfile("PtCorrOverGen_GenPt_E", "1.5<|eta|<3", log10PtBins, log10PtMin, log10PtMax, 0.8, 1.2, " ");
      mPtCorrOverGen_GenPt_F = ibooker.bookProfile("PtCorrOverGen_GenPt_F", "3<|eta|<6",   log10PtBins, log10PtMin, log10PtMax, 0.8, 1.2, " ");
      //if eta range changed here need change in JetTesterPostProcessor as well
      mPtCorrOverGen_GenEta_20_40    = ibooker.bookProfile("PtCorrOverGen_GenEta_20_40",      "20<genPt<40;#eta",    90, etaRange, 0.8, 1.2, " ");
      mPtCorrOverGen_GenEta_40_200    = ibooker.bookProfile("PtCorrOverGen_GenEta_40_200",    "40<genPt<200;#eta",    90, etaRange, 0.8, 1.2, " ");
      mPtCorrOverGen_GenEta_200_600   = ibooker.bookProfile("PtCorrOverGen_GenEta_200_600",   "200<genPt<600;#eta",   90, etaRange, 0.8, 1.2, " ");
      mPtCorrOverGen_GenEta_600_1500  = ibooker.bookProfile("PtCorrOverGen_GenEta_600_1500",  "600<genPt<1500;#eta",  90, etaRange, 0.8, 1.2, " ");
      mPtCorrOverGen_GenEta_1500_3500 = ibooker.bookProfile("PtCorrOverGen_GenEta_1500_3500", "1500<genPt<3500;#eta", 90, etaRange, 0.8, 1.2, " ");
      mPtCorrOverGen_GenEta_3500_5000 = ibooker.bookProfile("PtCorrOverGen_GenEta_3500_5000", "3500<genPt<5000;#eta", 90, etaRange, 0.8, 1.2, " ");
      mPtCorrOverGen_GenEta_5000_6500 = ibooker.bookProfile("PtCorrOverGen_GenEta_5000_6500", "5000<genPt<6500;#eta", 90, etaRange, 0.8, 1.2, " ");
      mPtCorrOverGen_GenEta_3500      = ibooker.bookProfile("PtCorrOverGen_GenEta_3500",      "genPt>3500;#eta",      90, etaRange, 0.8, 1.2, " ");
    }

    mGenEta      = ibooker.book1D("GenEta",      "GenEta",      120,   -6,    6);
    mGenPhi      = ibooker.book1D("GenPhi",      "GenPhi",       70, -3.5,  3.5);
    mGenPt       = ibooker.book1D("GenPt",       "GenPt",       100,    0,  1000);
    mGenEtaFirst = ibooker.book1D("GenEtaFirst", "GenEtaFirst", 120,   -6,    6);
    mGenPhiFirst = ibooker.book1D("GenPhiFirst", "GenPhiFirst",  70, -3.5,  3.5);
    mPtHat       = ibooker.book1D("PtHat",       "PtHat",       100,    0, 1000); 
    mDeltaEta    = ibooker.book1D("DeltaEta",    "DeltaEta",    100, -0.5,  0.5);
    mDeltaPhi    = ibooker.book1D("DeltaPhi",    "DeltaPhi",    100, -0.5,  0.5);
    mDeltaPt     = ibooker.book1D("DeltaPt",     "DeltaPt",     100, -1.0,  1.0);
    
    mPtRecoOverGen_B_20_40    = ibooker.book1D("PtRecoOverGen_B_20_40",    "20<genpt<40",    90, 0, 2);
    mPtRecoOverGen_E_20_40    = ibooker.book1D("PtRecoOverGen_E_20_40",    "20<genpt<40",    90, 0, 2);
    mPtRecoOverGen_F_20_40    = ibooker.book1D("PtRecoOverGen_F_20_40",    "20<genpt<40",    90, 0, 2);
    mPtRecoOverGen_B_40_200    = ibooker.book1D("PtRecoOverGen_B_40_200",    "40<genpt<200",    90, 0, 2);
    mPtRecoOverGen_E_40_200    = ibooker.book1D("PtRecoOverGen_E_40_200",    "40<genpt<200",    90, 0, 2);
    mPtRecoOverGen_F_40_200    = ibooker.book1D("PtRecoOverGen_F_40_200",    "40<genpt<200",    90, 0, 2);
    mPtRecoOverGen_B_200_600   = ibooker.book1D("PtRecoOverGen_B_200_600",   "200<genpt<600",   90, 0, 2);
    mPtRecoOverGen_E_200_600   = ibooker.book1D("PtRecoOverGen_E_200_600",   "200<genpt<600",   90, 0, 2);
    mPtRecoOverGen_F_200_600   = ibooker.book1D("PtRecoOverGen_F_200_600",   "200<genpt<600",   90, 0, 2);
    mPtRecoOverGen_B_600_1500  = ibooker.book1D("PtRecoOverGen_B_600_1500",  "600<genpt<1500",  90, 0, 2);
    mPtRecoOverGen_E_600_1500  = ibooker.book1D("PtRecoOverGen_E_600_1500",  "600<genpt<1500",  90, 0, 2);
    mPtRecoOverGen_F_600_1500  = ibooker.book1D("PtRecoOverGen_F_600_1500",  "600<genpt<1500",  90, 0, 2);
    mPtRecoOverGen_B_1500_3500 = ibooker.book1D("PtRecoOverGen_B_1500_3500", "1500<genpt<3500", 90, 0, 2);
    mPtRecoOverGen_E_1500_3500 = ibooker.book1D("PtRecoOverGen_E_1500_3500", "1500<genpt<3500", 90, 0, 2);
    mPtRecoOverGen_F_1500_3500 = ibooker.book1D("PtRecoOverGen_F_1500_3500", "1500<genpt<3500", 90, 0, 2);
    mPtRecoOverGen_B_3500_5000 = ibooker.book1D("PtRecoOverGen_B_3500_5000", "3500<genpt<5000", 90, 0, 2);
    mPtRecoOverGen_E_3500_5000 = ibooker.book1D("PtRecoOverGen_E_3500_5000", "3500<genpt<5000", 90, 0, 2);
    mPtRecoOverGen_B_5000_6500 = ibooker.book1D("PtRecoOverGen_B_5000_6500", "5000<genpt<6500", 90, 0, 2);
    mPtRecoOverGen_E_5000_6500 = ibooker.book1D("PtRecoOverGen_E_5000_6500", "5000<genpt<6500", 90, 0, 2);
    mPtRecoOverGen_B_3500      = ibooker.book1D("PtRecoOverGen_B_3500",      "genpt>3500",      90, 0, 2);
    mPtRecoOverGen_E_3500      = ibooker.book1D("PtRecoOverGen_E_3500",      "genpt>3500",      90, 0, 2);
    mPtRecoOverGen_F_3500      = ibooker.book1D("PtRecoOverGen_F_3500",      "genpt>3500",      90, 0, 2);

    // Generation profiles
    mPtRecoOverGen_GenPt_B          = ibooker.bookProfile("PtRecoOverGen_GenPt_B",          "0<|eta|<1.5",     log10PtBins, log10PtMin, log10PtMax, 0, 2, " ");
    mPtRecoOverGen_GenPt_E          = ibooker.bookProfile("PtRecoOverGen_GenPt_E",          "1.5<|eta|<3",     log10PtBins, log10PtMin, log10PtMax, 0, 2, " ");
    mPtRecoOverGen_GenPt_F          = ibooker.bookProfile("PtRecoOverGen_GenPt_F",          "3<|eta|<6",       log10PtBins, log10PtMin, log10PtMax, 0, 2, " ");
    mPtRecoOverGen_GenPhi_B         = ibooker.bookProfile("PtRecoOverGen_GenPhi_B",         "0<|eta|<1.5",     70, -3.5, 3.5, 0, 2, " ");
    mPtRecoOverGen_GenPhi_E         = ibooker.bookProfile("PtRecoOverGen_GenPhi_E",         "1.5<|eta|<3",     70, -3.5, 3.5, 0, 2, " ");
    mPtRecoOverGen_GenPhi_F         = ibooker.bookProfile("PtRecoOverGen_GenPhi_F",         "3<|eta|<6",       70, -3.5, 3.5, 0, 2, " ");
    //if eta range changed here need change in JetTesterPostProcessor as well
    mPtRecoOverGen_GenEta_20_40    = ibooker.bookProfile("PtRecoOverGen_GenEta_20_40",    "20<genpt<40",    90, etaRange, 0, 2, " ");
    mPtRecoOverGen_GenEta_40_200    = ibooker.bookProfile("PtRecoOverGen_GenEta_40_200",    "40<genpt<200",    90, etaRange, 0, 2, " ");
    mPtRecoOverGen_GenEta_200_600   = ibooker.bookProfile("PtRecoOverGen_GenEta_200_600",   "200<genpt<600",   90, etaRange, 0, 2, " ");
    mPtRecoOverGen_GenEta_600_1500  = ibooker.bookProfile("PtRecoOverGen_GenEta_600_1500",  "600<genpt<1500",  90, etaRange, 0, 2, " ");
    mPtRecoOverGen_GenEta_1500_3500 = ibooker.bookProfile("PtRecoOverGen_GenEta_1500_3500", "1500<genpt<3500", 90, etaRange, 0, 2, " ");
    mPtRecoOverGen_GenEta_3500_5000 = ibooker.bookProfile("PtRecoOverGen_GenEta_3500_5000", "3500<genpt<5000", 90, etaRange, 0, 2, " ");
    mPtRecoOverGen_GenEta_5000_6500 = ibooker.bookProfile("PtRecoOverGen_GenEta_5000_6500", "5000<genpt<6500", 90, etaRange, 0, 2, " ");
    mPtRecoOverGen_GenEta_3500      = ibooker.bookProfile("PtRecoOverGen_GenEta_3500",      "genpt>3500",      90, etaRange, 0, 2, " "); 
  
    // Some jet algebra
    //------------------------------------------------------------------------
    mEtaFirst   = ibooker.book1D("EtaFirst",   "EtaFirst",   120,   -6,    6); 
    mPhiFirst   = ibooker.book1D("PhiFirst",   "PhiFirst",    70, -3.5,  3.5);      
    mPtFirst    = ibooker.book1D("PtFirst",    "PtFirst",    50,    0,  1000); 
    mMjj        = ibooker.book1D("Mjj",        "Mjj",        100,    0, 2000); 
    mNJetsEta_B_20_40 = ibooker.book1D("NJetsEta_B_20_40", "NJetsEta_B 20<Pt<40",  15,    0,   15);
    mNJetsEta_E_20_40 = ibooker.book1D("NJetsEta_E_20_40", "NJetsEta_E 20<Pt<40",  15,    0,   15);
    mNJetsEta_B_40 = ibooker.book1D("NJetsEta_B", "NJetsEta_B 40<Pt",  15,    0,   15);
    mNJetsEta_E_40 = ibooker.book1D("NJetsEta_E", "NJetsEta_E 40<Pt",  15,    0,   15);
    mNJets_40 = ibooker.book1D("NJets", "NJets 40>Pt",  15,    0,   15);
    mNJets1 = ibooker.bookProfile("NJets1", "Number of jets above Pt threshold", 100, 0,  200, 100, 0, 50, "s");
    mNJets2 = ibooker.bookProfile("NJets2", "Number of jets above Pt threshold", 100, 0, 4000, 100, 0, 50, "s");


    if (isCaloJet) {
      maxEInEmTowers              = ibooker.book1D("maxEInEmTowers", "maxEInEmTowers", 50,0,500);
      maxEInHadTowers             = ibooker.book1D("maxEInHadTowers", "maxEInHadTowers", 50,0,500);
      energyFractionHadronic      = ibooker.book1D("energyFractionHadronic", "energyFractionHadronic", 50,0,1);
      emEnergyFraction            = ibooker.book1D("emEnergyFraction", "emEnergyFraction", 50,0,1);
      hadEnergyInHB               = ibooker.book1D("hadEnergyInHB", "hadEnergyInHB", 50,0,500);
      hadEnergyInHO               = ibooker.book1D("hadEnergyInHO", "hadEnergyInHO", 50,0,500);
      hadEnergyInHE               = ibooker.book1D("hadEnergyInHE", "hadEnergyInHE", 50,0,500);
      hadEnergyInHF               = ibooker.book1D("hadEnergyInHF", "hadEnergyInHF", 50,0,500);
      emEnergyInEB                = ibooker.book1D("emEnergyInEB", "emEnergyInEB", 50,0,500);
      emEnergyInEE                = ibooker.book1D("emEnergyInEE", "emEnergyInEE", 50,0,500);
      emEnergyInHF                = ibooker.book1D("emEnergyInHF", "emEnergyInHF", 50,0,500);
      towersArea                  = ibooker.book1D("towersArea", "towersArea", 50,0,1);
      n90                         = ibooker.book1D("n90", "n90", 30,0,30);
      n60                         = ibooker.book1D("n60", "n60", 30,0,30);
    }

    if (isPFJet || isMiniAODJet) {
      muonMultiplicity = ibooker.book1D("muonMultiplicity", "muonMultiplicity", 10,0,10);
      chargedMultiplicity = ibooker.book1D("chargedMultiplicity", "chargedMultiplicity", 100,0,100);
      chargedEmEnergy = ibooker.book1D("chargedEmEnergy", "chargedEmEnergy", 100,0,500);
      neutralEmEnergy = ibooker.book1D("neutralEmEnergy", "neutralEmEnergy", 100,0,500);
      chargedHadronEnergy = ibooker.book1D("chargedHadronEnergy", "chargedHadronEnergy", 100,0,500);
      neutralHadronEnergy = ibooker.book1D("neutralHadronEnergy", "neutralHadronEnergy", 100,0,500);
      chargedHadronEnergyFraction = ibooker.book1D("chargedHadronEnergyFraction", "chargedHadronEnergyFraction", 50,0,1);
      neutralHadronEnergyFraction = ibooker.book1D("neutralHadronEnergyFraction", "neutralHadronEnergyFraction", 50,0,1);
      chargedEmEnergyFraction = ibooker.book1D("chargedEmEnergyFraction", "chargedEmEnergyFraction", 50,0,1);
      neutralEmEnergyFraction = ibooker.book1D("neutralEmEnergyFraction", "neutralEmEnergyFraction", 50,0,1);
      photonEnergy = ibooker.book1D("photonEnergy", "photonEnergy", 50,0,500);
      photonEnergyFraction = ibooker.book1D("photonEnergyFraction", "photonEnergyFraction", 50,0,1);
      electronEnergy = ibooker.book1D("electronEnergy", "electronEnergy", 50,0,500);
      electronEnergyFraction = ibooker.book1D("electronEnergyFraction", "electronEnergyFraction", 50,0,1);
      muonEnergy = ibooker.book1D("muonEnergy", "muonEnergy", 50,0,500);
      muonEnergyFraction = ibooker.book1D("muonEnergyFraction", "muonEnergyFraction", 50,0,1);
      HFHadronEnergy = ibooker.book1D("HFHadronEnergy", "HFHadronEnergy", 50,0,500);
      HFHadronEnergyFraction = ibooker.book1D("HFHadronEnergyFraction", "HFHadronEnergyFraction", 50,0,1);
      HFEMEnergy = ibooker.book1D("HFEMEnergy", "HFEMEnergy", 50,0,500);
      HFEMEnergyFraction = ibooker.book1D("HFEMEnergyFraction", "HFEMEnergyFraction", 50,0,1);
      chargedHadronMultiplicity = ibooker.book1D("chargedHadronMultiplicity", "chargedHadronMultiplicity", 50,0,50);
      neutralHadronMultiplicity = ibooker.book1D("neutralHadronMultiplicity", "neutralHadronMultiplicity", 50,0,50);
      photonMultiplicity = ibooker.book1D("photonMultiplicity", "photonMultiplicity", 10,0,10);
      electronMultiplicity = ibooker.book1D("electronMultiplicity", "electronMultiplicity", 10,0,10);
      HFHadronMultiplicity = ibooker.book1D("HFHadronMultiplicity", "HFHadronMultiplicity", 50,0,50);
      HFEMMultiplicity = ibooker.book1D("HFEMMultiplicity", "HFEMMultiplicity", 50,0,50);
      chargedMuEnergy = ibooker.book1D("chargedMuEnergy", "chargedMuEnergy", 50,0,500);
      chargedMuEnergyFraction = ibooker.book1D("chargedMuEnergyFraction", "chargedMuEnergyFraction", 50,0,1);
      neutralMultiplicity = ibooker.book1D("neutralMultiplicity", "neutralMultiplicity", 50,0,50);
      HOEnergy = ibooker.book1D("HOEnergy", "HOEnergy", 50,0,500);
      HOEnergyFraction = ibooker.book1D("HOEnergyFraction", "HOEnergyFraction", 50,0,1);
    }
}


//------------------------------------------------------------------------------
// ~JetTester
//------------------------------------------------------------------------------
JetTester::~JetTester() {}


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
  edm::Handle<PFJetCollection>   pfJets;
//  edm::Handle<JPTJetCollection>  jptJets;
  edm::Handle<pat::JetCollection> patJets;

  if (isCaloJet) mEvent.getByToken(caloJetsToken_, caloJets);
  if (isPFJet)   mEvent.getByToken(pfJetsToken_, pfJets);
//  if (isJPTJet)  mEvent.getByToken(jptJetsToken_, jptJets);
  if (isMiniAODJet)   mEvent.getByToken(patJetsToken_, patJets);

  if (isCaloJet && !caloJets.isValid()) return;
  if (isPFJet   && !pfJets.isValid())   return;
//  if (isJPTJet  && !jptJets.isValid())  return;
 if (isMiniAODJet   && !patJets.isValid())   return;


  if (isCaloJet)
    {
      for (unsigned ijet=0; ijet<caloJets->size(); ijet++)
	recoJets.push_back((*caloJets)[ijet]);
    }

/*  if (isJPTJet)
    {
      for (unsigned ijet=0; ijet<jptJets->size(); ijet++)
	recoJets.push_back((*jptJets)[ijet]);
    }*/

  if (isPFJet) {
    for (unsigned ijet=0; ijet<pfJets->size(); ijet++)
      recoJets.push_back((*pfJets)[ijet]);
  }
  if (isMiniAODJet) {
    for (unsigned ijet=0; ijet<patJets->size(); ijet++)
      recoJets.push_back((*patJets)[ijet]);
  }

  int nJet      = 0;
  int nJet_E_20_40 = 0;
  int nJet_B_20_40 = 0;
  int nJet_E_40 = 0;
  int nJet_B_40 = 0;
  int nJet_40 = 0;

  int index_first_jet=-1;
  double pt_first=-1;

 int index_second_jet=-1;
 double pt_second=-1;

  for (unsigned ijet=0; ijet<recoJets.size(); ijet++) {
    bool pass_lowjet=false;
    bool pass_mediumjet = false;
    if(!isMiniAODJet){
      if (  (recoJets[ijet].pt() > 20.) &&  (recoJets[ijet].pt() < mRecoJetPtThreshold)) {
	pass_lowjet=true;
      }
    }
    if(isMiniAODJet){
      if((recoJets[ijet].pt()*(*patJets)[ijet].jecFactor("Uncorrected"))>20. && ((recoJets[ijet].pt()*(*patJets)[ijet].jecFactor("Uncorrected")) < mRecoJetPtThreshold)){
	pass_lowjet=true;
      }
    }
    if (pass_lowjet) {
      if (fabs(recoJets[ijet].eta()) > 1.5)
        nJet_E_20_40++;
      else
        nJet_B_20_40++;	  
    }
    if(!isMiniAODJet){
      if (recoJets[ijet].pt() > mRecoJetPtThreshold) {
	pass_mediumjet = true;
      }
    }
    if(isMiniAODJet){
      if((recoJets[ijet].pt()*(*patJets)[ijet].jecFactor("Uncorrected")) > mRecoJetPtThreshold){
	pass_mediumjet=true;
      }
    }
    if (pass_mediumjet) {
      if(isMiniAODJet){
	if( (recoJets[ijet].pt()*(*patJets)[ijet].jecFactor("Uncorrected"))>pt_first){
	  pt_second=pt_first;
	  pt_first=recoJets[ijet].pt()*(*patJets)[ijet].jecFactor("Uncorrected");
	  index_second_jet=index_first_jet;
	  index_first_jet=ijet;
	}else if( (recoJets[ijet].pt()*(*patJets)[ijet].jecFactor("Uncorrected"))>pt_second){
	  index_second_jet=ijet;
	  pt_second=recoJets[ijet].pt()*(*patJets)[ijet].jecFactor("Uncorrected");
	}
      }
      //counting forward and barrel jets
      if (fabs(recoJets[ijet].eta()) > 1.5)
        nJet_E_40++;
      else
        nJet_B_40++;	  
      nJet_40++;

      if (mEta) mEta->Fill(recoJets[ijet].eta());

      if (mJetArea)      mJetArea     ->Fill(recoJets[ijet].jetArea());
      if (mPhi)          mPhi         ->Fill(recoJets[ijet].phi());
      if(!isMiniAODJet){
	if (mEnergy)       mEnergy      ->Fill(recoJets[ijet].energy());
	if (mP)            mP           ->Fill(recoJets[ijet].p());
	if (mPt)           mPt          ->Fill(recoJets[ijet].pt());
	if (mMass)         mMass        ->Fill(recoJets[ijet].mass());
      }else{
	if (mEnergy)       mEnergy      ->Fill(recoJets[ijet].energy()*(*patJets)[ijet].jecFactor("Uncorrected"));
	if (mP)            mP           ->Fill(recoJets[ijet].p()*(*patJets)[ijet].jecFactor("Uncorrected"));
	if (mPt)           mPt          ->Fill(recoJets[ijet].pt()*(*patJets)[ijet].jecFactor("Uncorrected"));
	if (mMass)         mMass        ->Fill(recoJets[ijet].mass()*(*patJets)[ijet].jecFactor("Uncorrected"));
      }
      if (mConstituents) mConstituents->Fill(recoJets[ijet].nConstituents());
      if(!isMiniAODJet){
	if (ijet == 0) {
	  if (mEtaFirst) mEtaFirst->Fill(recoJets[ijet].eta());
	  if (mPhiFirst) mPhiFirst->Fill(recoJets[ijet].phi());
	  if (mPtFirst)  mPtFirst ->Fill(recoJets[ijet].pt());
	}

	if (ijet == 0) {nJet++; p4tmp[0] = recoJets[ijet].p4();}
	if (ijet == 1) {nJet++; p4tmp[1] = recoJets[ijet].p4();}
      }
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
      // ---- PF Jet specific information ----
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
        HOEnergy ->Fill((*pfJets)[ijet].hoEnergy());
        HOEnergyFraction ->Fill((*pfJets)[ijet].hoEnergyFraction());
     }
   if (isMiniAODJet && (*patJets)[ijet].isPFJet()) {
        muonMultiplicity ->Fill((*patJets)[ijet].muonMultiplicity());
        chargedMultiplicity ->Fill((*patJets)[ijet].chargedMultiplicity());
        chargedEmEnergy ->Fill((*patJets)[ijet].chargedEmEnergy());
        neutralEmEnergy ->Fill((*patJets)[ijet].neutralEmEnergy());
        chargedHadronEnergy ->Fill((*patJets)[ijet].chargedHadronEnergy());
        neutralHadronEnergy ->Fill((*patJets)[ijet].neutralHadronEnergy());
        chargedHadronEnergyFraction ->Fill((*patJets)[ijet].chargedHadronEnergyFraction());
        neutralHadronEnergyFraction ->Fill((*patJets)[ijet].neutralHadronEnergyFraction());
        chargedEmEnergyFraction ->Fill((*patJets)[ijet].chargedEmEnergyFraction());
        neutralEmEnergyFraction ->Fill((*patJets)[ijet].neutralEmEnergyFraction());
        photonEnergy ->Fill((*patJets)[ijet].photonEnergy());
        photonEnergyFraction ->Fill((*patJets)[ijet].photonEnergyFraction());
        electronEnergy ->Fill((*patJets)[ijet].electronEnergy());
        electronEnergyFraction ->Fill((*patJets)[ijet].electronEnergyFraction());
        muonEnergy ->Fill((*patJets)[ijet].muonEnergy());
        muonEnergyFraction ->Fill((*patJets)[ijet].muonEnergyFraction());
        HFHadronEnergy ->Fill((*patJets)[ijet].HFHadronEnergy());
        HFHadronEnergyFraction ->Fill((*patJets)[ijet].HFHadronEnergyFraction());
        HFEMEnergy ->Fill((*patJets)[ijet].HFEMEnergy());
        HFEMEnergyFraction ->Fill((*patJets)[ijet].HFEMEnergyFraction());
        chargedHadronMultiplicity ->Fill((*patJets)[ijet].chargedHadronMultiplicity());
        neutralHadronMultiplicity ->Fill((*patJets)[ijet].neutralHadronMultiplicity());
        photonMultiplicity ->Fill((*patJets)[ijet].photonMultiplicity());
        electronMultiplicity ->Fill((*patJets)[ijet].electronMultiplicity());
        HFHadronMultiplicity ->Fill((*patJets)[ijet].HFHadronMultiplicity());
        HFEMMultiplicity ->Fill((*patJets)[ijet].HFEMMultiplicity());
        chargedMuEnergy ->Fill((*patJets)[ijet].chargedMuEnergy());
        chargedMuEnergyFraction ->Fill((*patJets)[ijet].chargedMuEnergyFraction());
        neutralMultiplicity ->Fill((*patJets)[ijet].neutralMultiplicity()); 
        HOEnergy ->Fill((*patJets)[ijet].hoEnergy());
        HOEnergyFraction ->Fill((*patJets)[ijet].hoEnergyFraction());
      }
    }//fill quantities for medium jets
  }

  if (mNJetsEta_B_20_40) mNJetsEta_B_20_40->Fill(nJet_B_20_40);
  if (mNJetsEta_E_20_40) mNJetsEta_E_20_40->Fill(nJet_E_20_40);
  if (mNJetsEta_B_40) mNJetsEta_B_40->Fill(nJet_B_40);
  if (mNJetsEta_E_40) mNJetsEta_E_40->Fill(nJet_E_40);
  if (mNJets_40) mNJets_40->Fill(nJet_40); 
  if(!isMiniAODJet){
    if (nJet >= 2)
      {
	if (mMjj) mMjj->Fill((p4tmp[0]+p4tmp[1]).mass());
      }
  }else{
    if(index_first_jet>-1){
      if (mEtaFirst) mEtaFirst->Fill(recoJets[index_first_jet].eta());
      if (mPhiFirst) mPhiFirst->Fill(recoJets[index_first_jet].phi());
      if (mPtFirst)  mPtFirst ->Fill(recoJets[index_first_jet].pt()*(*patJets)[index_first_jet].jecFactor("Uncorrected"));
      nJet++; p4tmp[0] = recoJets[index_first_jet].p4()*(*patJets)[index_first_jet].jecFactor("Uncorrected");
    }
    if(index_second_jet>-1){
      nJet++; p4tmp[1] = recoJets[index_second_jet].p4()*(*patJets)[index_second_jet].jecFactor("Uncorrected");
    }
    if (nJet >= 2)
      {
	if (mMjj) mMjj->Fill((p4tmp[0]+p4tmp[1]).mass());
      }
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
	if(!isMiniAODJet){
	  if (recoJets[ijet].pt() > ptStep1) njets1++;
	  if (recoJets[ijet].pt() > ptStep2) njets2++;
	}else{
	  if ((recoJets[ijet].pt()*(*patJets)[ijet].jecFactor("Uncorrected")) > ptStep1) njets1++;
	  if ((recoJets[ijet].pt()*(*patJets)[ijet].jecFactor("Uncorrected")) > ptStep2) njets2++;
	}
	mNJets1->Fill(ptStep1, njets1);
	mNJets2->Fill(ptStep2, njets2);
      }
    }


  // Corrected jets
  //----------------------------------------------------------------------------
  double scale = -999;
  edm::Handle<reco::JetCorrector> jetCorr;
  bool pass_correction_flag =false;
  if(!isMiniAODJet && !mJetCorrector.label().empty()){
    mEvent.getByToken(jetCorrectorToken_, jetCorr);
    if (jetCorr.isValid()){
      pass_correction_flag=true;
    }
  }
  if(isMiniAODJet){
    pass_correction_flag =true;
  }
  for (unsigned ijet=0; ijet<recoJets.size(); ijet++) {
    Jet correctedJet = recoJets[ijet];
    if(pass_correction_flag){
      if (isCaloJet) scale = jetCorr->correction((*caloJets)[ijet]); 
      if (isPFJet)   scale = jetCorr->correction((*pfJets)[ijet]); 
      //if (isJPTJet)  scale = jetCorr->correction((*jptJets)[ijet]);
      if(!isMiniAODJet){
	correctedJet.scaleEnergy(scale); 
      }
      
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
      if(isMiniAODJet){
	ijetPt=recoJets[ijet].pt()*(*patJets)[ijet].jecFactor("Uncorrected");
      }
      double ratio   = correctedJet.pt() / ijetPt;
      if(isMiniAODJet){
	ratio =1./(*patJets)[ijet].jecFactor("Uncorrected");
      }
      
      if      (fabs(ijetEta) < 1.5) mPtCorrOverReco_Pt_B->Fill(log10(ijetPt), ratio);
      else if (fabs(ijetEta) < 3.0) mPtCorrOverReco_Pt_E->Fill(log10(ijetPt), ratio);
      else if (fabs(ijetEta) < 6.0) mPtCorrOverReco_Pt_F->Fill(log10(ijetPt), ratio);
      
      if      (ijetPt <  40) mPtCorrOverReco_Eta_20_40   ->Fill(ijetEta, ratio);
      else if (ijetPt <  200) mPtCorrOverReco_Eta_40_200  ->Fill(ijetEta, ratio);
      else if (ijetPt <  600) mPtCorrOverReco_Eta_200_600  ->Fill(ijetEta, ratio);
      else if (ijetPt < 1500) mPtCorrOverReco_Eta_600_1500 ->Fill(ijetEta, ratio);
      else if (ijetPt < 3500) mPtCorrOverReco_Eta_1500_3500->Fill(ijetEta, ratio);
      else if (ijetPt < 5000) mPtCorrOverReco_Eta_3500_5000->Fill(ijetEta, ratio);
      else if (ijetPt < 6500) mPtCorrOverReco_Eta_5000_6500->Fill(ijetEta, ratio);
      if (ijetPt > 3500) mPtCorrOverReco_Eta_3500->Fill(ijetEta, ratio);
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
      edm::Handle<GenEventInfoProduct> myGenEvt;
      mEvent.getByToken(evtToken_, myGenEvt);

      if (myGenEvt.isValid()) {
        if(myGenEvt->hasBinningValues()){
	  double ptHat = myGenEvt->binningValues()[0];
	  if (mPtHat) mPtHat->Fill(ptHat);
        }
      }
      // Gen jets
      //------------------------------------------------------------------------
      edm::Handle<GenJetCollection> genJets;
      mEvent.getByToken(genJetsToken_, genJets);

      if (!genJets.isValid()) return;
      
      for (GenJetCollection::const_iterator gjet=genJets->begin();  gjet!=genJets->end(); gjet++)	{
	//for MiniAOD we have here intrinsic thresholds, introduce also threshold for RECO
	if(gjet->pt() > mMatchGenPtThreshold){
	  if (mGenEta) mGenEta->Fill(gjet->eta());
	  if (mGenPhi) mGenPhi->Fill(gjet->phi());
	  if (mGenPt)  mGenPt ->Fill(gjet->pt());
	  if (gjet == genJets->begin()) {
	    if (mGenEtaFirst) mGenEtaFirst->Fill(gjet->eta());
	    if (mGenPhiFirst) mGenPhiFirst->Fill(gjet->phi());
	  }
	}
      }

      if (!(mInputGenCollection.label().empty())) {
      for (GenJetCollection::const_iterator gjet=genJets->begin(); gjet!=genJets->end(); gjet++) {
        if (fabs(gjet->eta()) > 6.) continue;  // Out of the detector 
        if (gjet->pt() < mMatchGenPtThreshold) continue;
        if (recoJets.size() <= 0) continue;
        // pt response
        //------------------------------------------------------------
	int iMatch    =   -1;
	double CorrdeltaRBest = 999;
	double CorrJetPtBest  =   0;
	for (unsigned ijet=0; ijet<recoJets.size(); ++ijet) {
	  Jet correctedJet = recoJets[ijet];
	  if(pass_correction_flag && !isMiniAODJet){
	    if (isCaloJet) scale = jetCorr->correction((*caloJets)[ijet]); 
	    if (isPFJet)   scale = jetCorr->correction((*pfJets)[ijet]); 
	    correctedJet.scaleEnergy(scale);
	  }
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
	if(!isMiniAODJet){
	    fillMatchHists(gjet->eta(),  gjet->phi(),  gjet->pt(), recoJets[iMatch].eta(), recoJets[iMatch].phi(),  recoJets[iMatch].pt());
	  }else{
	  fillMatchHists(gjet->eta(),  gjet->phi(),  gjet->pt(), (*patJets)[iMatch].eta(), (*patJets)[iMatch].phi(),(*patJets)[iMatch].pt()*(*patJets)[iMatch].jecFactor("Uncorrected"));
	}
        if (pass_correction_flag) {//fill only for corrected jets
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
              else if (gjet->pt() < 5000) mPtCorrOverGen_GenEta_3500_5000->Fill(gjet->eta(), response);
              else if (gjet->pt() < 6500) mPtCorrOverGen_GenEta_5000_6500->Fill(gjet->eta(), response);
              if (gjet->pt() > 3500) mPtCorrOverGen_GenEta_3500->Fill(gjet->eta(), response);
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
      else if (GenPt < 5000)         mPtRecoOverGen_B_3500_5000->Fill(RecoPt / GenPt);
      else if (GenPt < 6500)         mPtRecoOverGen_B_5000_6500->Fill(RecoPt / GenPt);
      if (GenPt>3500)         mPtRecoOverGen_B_3500->Fill(RecoPt / GenPt);
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
      else if (GenPt < 5000)         mPtRecoOverGen_E_3500_5000->Fill(RecoPt / GenPt);
      else if (GenPt < 6500)         mPtRecoOverGen_E_5000_6500->Fill(RecoPt / GenPt);
      if (GenPt>3500)         mPtRecoOverGen_E_3500->Fill(RecoPt / GenPt);
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
      if (GenPt>3500)                mPtRecoOverGen_F_3500->Fill(RecoPt / GenPt);
    }

  if (GenPt > 20 && GenPt < 40) mPtRecoOverGen_GenEta_20_40   ->Fill(GenEta, RecoPt / GenPt);
  else if (GenPt <  200)         mPtRecoOverGen_GenEta_40_200  ->Fill(GenEta, RecoPt / GenPt);
  else if (GenPt <  600)         mPtRecoOverGen_GenEta_200_600  ->Fill(GenEta, RecoPt / GenPt);
  else if (GenPt < 1500)         mPtRecoOverGen_GenEta_600_1500 ->Fill(GenEta, RecoPt / GenPt);
  else if (GenPt < 3500)         mPtRecoOverGen_GenEta_1500_3500->Fill(GenEta, RecoPt / GenPt);
  else if (GenPt < 5000)         mPtRecoOverGen_GenEta_3500_5000->Fill(GenEta, RecoPt / GenPt);
  else if (GenPt < 6500)         mPtRecoOverGen_GenEta_5000_6500->Fill(GenEta, RecoPt / GenPt);
  if (GenPt > 3500)              mPtRecoOverGen_GenEta_3500->Fill(GenEta, RecoPt / GenPt);
}
