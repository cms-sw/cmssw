#ifndef ValidationRecoJetsJetTester_HeavyIons_h
#define ValidationRecoJetsJetTester_HeavyIons_h

// Producer for validation histograms for Calo, JPT and PF jet objects
// F. Ratnikov, Sept. 7, 2006
// Modified by Chiyoung Jeong, Feb. 2, 2010
// Modified by J. Piedra, Sept. 11, 2013
// Rewritten by Viola Sordini, Matthias Artur Weber, Robert Schoefbeck Nov./Dez.
// 2013 Modified by Raghav Kunnawalkam Elayavalli, Aug 18th 2014 to run in 72X
//                                          , Oct 22nd 2014 to run in 73X
//                                          , Dec 10th 2014 74X and adding the
//                                          PF candidates information to easily
//                                          detect
//                                                          the voronoi
//                                                          subtraction
//                                                          algorithm failure
//                                                          modes.

#include <cmath>
#include <string>

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/CandidateWithRef.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/JPTJet.h"
#include "DataFormats/JetReco/interface/JPTJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"

// include the pf candidates
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
// include the voronoi subtraction
#include "DataFormats/HeavyIonEvent/interface/VoronoiBackground.h"
#include "RecoHI/HiJetAlgos/interface/UEParameters.h"
// include the centrality variables
#include "DataFormats/HeavyIonEvent/interface/Centrality.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "RecoJets/JetProducers/interface/JetMatchingTools.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

const Int_t MAXPARTICLE = 10000;
const Double_t BarrelEta = 2.0;
const Double_t EndcapEta = 3.0;
const Double_t ForwardEta = 5.0;

class JetTester_HeavyIons : public DQMEDAnalyzer {
public:
  explicit JetTester_HeavyIons(const edm::ParameterSet &);
  ~JetTester_HeavyIons() override;

  void analyze(const edm::Event &, const edm::EventSetup &) override;
  // virtual void beginJob();
  // virtual void endJob();
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  // reco::Vertex::Point getVtx(const edm::Event& ev);

  // double getEt(const DetID )

private:
  void fillMatchHists(const double GenEta,
                      const double GenPhi,
                      const double GenPt,
                      const double RecoEta,
                      const double RecoPhi,
                      const double RecoPt);

  edm::InputTag mInputCollection;
  edm::InputTag mInputGenCollection;
  edm::InputTag mInputPFCandCollection;
  // edm::InputTag   mInputCandCollection;
  // edm::InputTag   rhoTag;
  edm::InputTag centralityTag_;
  edm::EDGetTokenT<reco::Centrality> centralityToken;
  edm::Handle<reco::Centrality> centrality_;

  edm::InputTag centralityBinTag_;
  edm::EDGetTokenT<int> centralityBinToken;
  edm::Handle<int> centralityBin_;

  std::string mOutputFile;
  std::string JetType;
  std::string UEAlgo;
  edm::InputTag Background;
  double mRecoJetPtThreshold;
  double mMatchGenPtThreshold;
  double mGenEnergyFractionThreshold;
  double mReverseEnergyFractionThreshold;
  double mRThreshold;
  std::string JetCorrectionService;

  // Tokens
  edm::EDGetTokenT<std::vector<reco::Vertex>> pvToken_;
  edm::EDGetTokenT<CaloTowerCollection> caloTowersToken_;
  edm::EDGetTokenT<reco::CaloJetCollection> caloJetsToken_;
  edm::EDGetTokenT<reco::PFJetCollection> pfJetsToken_;
  edm::EDGetTokenT<reco::BasicJetCollection> basicJetsToken_;
  edm::EDGetTokenT<reco::JPTJetCollection> jptJetsToken_;
  edm::EDGetTokenT<reco::GenJetCollection> genJetsToken_;
  edm::EDGetTokenT<GenEventInfoProduct> evtToken_;
  edm::EDGetTokenT<reco::PFCandidateCollection> pfCandToken_;
  edm::EDGetTokenT<reco::CandidateView> pfCandViewToken_;
  edm::EDGetTokenT<reco::CandidateView> caloCandViewToken_;
  // edm::EDGetTokenT<reco::VoronoiMap> backgrounds_;
  edm::EDGetTokenT<edm::ValueMap<reco::VoronoiBackground>> backgrounds_;
  edm::EDGetTokenT<std::vector<float>> backgrounds_value_;
  edm::EDGetTokenT<std::vector<reco::Vertex>> hiVertexToken_;

  // Include Particle flow variables
  MonitorElement *mNPFpart;
  MonitorElement *mPFPt;
  MonitorElement *mPFEta;
  MonitorElement *mPFPhi;
  MonitorElement *mPFVsPt;
  MonitorElement *mPFVsPtInitial;
  MonitorElement *mPFArea;
  MonitorElement *mNCalopart;
  MonitorElement *mCaloPt;
  MonitorElement *mCaloEta;
  MonitorElement *mCaloPhi;
  MonitorElement *mCaloVsPt;
  MonitorElement *mCaloVsPtInitial;
  MonitorElement *mCaloArea;
  MonitorElement *mSumpt;

  MonitorElement *mSumPFVsPt;
  MonitorElement *mSumPFVsPtInitial;
  MonitorElement *mSumPFPt;

  MonitorElement *mSumCaloVsPt;
  MonitorElement *mSumCaloVsPtInitial;
  MonitorElement *mSumCaloPt;

  MonitorElement *mSumSquaredPFVsPt;
  MonitorElement *mSumSquaredPFVsPtInitial;
  MonitorElement *mSumSquaredPFPt;

  MonitorElement *mSumSquaredCaloVsPt;
  MonitorElement *mSumSquaredCaloVsPtInitial;
  MonitorElement *mSumSquaredCaloPt;

  // Event variables (including centrality)
  MonitorElement *mNvtx;
  MonitorElement *mHF;

  // new additions Jan 12th 2015
  MonitorElement *mSumPFVsPt_HF;
  MonitorElement *mSumPFVsPtInitial_HF;
  MonitorElement *mSumPFPt_HF;
  MonitorElement *mDeltapT;
  MonitorElement *mDeltapT_eta;

  MonitorElement *mSumCaloVsPt_HF;
  MonitorElement *mSumCaloVsPtInitial_HF;
  MonitorElement *mSumCaloPt_HF;

  MonitorElement *mSumPFVsPtInitial_n5p191_n2p650;
  MonitorElement *mSumPFVsPtInitial_n2p650_n2p043;
  MonitorElement *mSumPFVsPtInitial_n2p043_n1p740;
  MonitorElement *mSumPFVsPtInitial_n1p740_n1p479;
  MonitorElement *mSumPFVsPtInitial_n1p479_n1p131;
  MonitorElement *mSumPFVsPtInitial_n1p131_n0p783;
  MonitorElement *mSumPFVsPtInitial_n0p783_n0p522;
  MonitorElement *mSumPFVsPtInitial_n0p522_0p522;
  MonitorElement *mSumPFVsPtInitial_0p522_0p783;
  MonitorElement *mSumPFVsPtInitial_0p783_1p131;
  MonitorElement *mSumPFVsPtInitial_1p131_1p479;
  MonitorElement *mSumPFVsPtInitial_1p479_1p740;
  MonitorElement *mSumPFVsPtInitial_1p740_2p043;
  MonitorElement *mSumPFVsPtInitial_2p043_2p650;
  MonitorElement *mSumPFVsPtInitial_2p650_5p191;

  MonitorElement *mSumPFVsPt_n5p191_n2p650;
  MonitorElement *mSumPFVsPt_n2p650_n2p043;
  MonitorElement *mSumPFVsPt_n2p043_n1p740;
  MonitorElement *mSumPFVsPt_n1p740_n1p479;
  MonitorElement *mSumPFVsPt_n1p479_n1p131;
  MonitorElement *mSumPFVsPt_n1p131_n0p783;
  MonitorElement *mSumPFVsPt_n0p783_n0p522;
  MonitorElement *mSumPFVsPt_n0p522_0p522;
  MonitorElement *mSumPFVsPt_0p522_0p783;
  MonitorElement *mSumPFVsPt_0p783_1p131;
  MonitorElement *mSumPFVsPt_1p131_1p479;
  MonitorElement *mSumPFVsPt_1p479_1p740;
  MonitorElement *mSumPFVsPt_1p740_2p043;
  MonitorElement *mSumPFVsPt_2p043_2p650;
  MonitorElement *mSumPFVsPt_2p650_5p191;

  MonitorElement *mSumPFPt_n5p191_n2p650;
  MonitorElement *mSumPFPt_n2p650_n2p043;
  MonitorElement *mSumPFPt_n2p043_n1p740;
  MonitorElement *mSumPFPt_n1p740_n1p479;
  MonitorElement *mSumPFPt_n1p479_n1p131;
  MonitorElement *mSumPFPt_n1p131_n0p783;
  MonitorElement *mSumPFPt_n0p783_n0p522;
  MonitorElement *mSumPFPt_n0p522_0p522;
  MonitorElement *mSumPFPt_0p522_0p783;
  MonitorElement *mSumPFPt_0p783_1p131;
  MonitorElement *mSumPFPt_1p131_1p479;
  MonitorElement *mSumPFPt_1p479_1p740;
  MonitorElement *mSumPFPt_1p740_2p043;
  MonitorElement *mSumPFPt_2p043_2p650;
  MonitorElement *mSumPFPt_2p650_5p191;

  MonitorElement *mSumCaloVsPtInitial_n5p191_n2p650;
  MonitorElement *mSumCaloVsPtInitial_n2p650_n2p043;
  MonitorElement *mSumCaloVsPtInitial_n2p043_n1p740;
  MonitorElement *mSumCaloVsPtInitial_n1p740_n1p479;
  MonitorElement *mSumCaloVsPtInitial_n1p479_n1p131;
  MonitorElement *mSumCaloVsPtInitial_n1p131_n0p783;
  MonitorElement *mSumCaloVsPtInitial_n0p783_n0p522;
  MonitorElement *mSumCaloVsPtInitial_n0p522_0p522;
  MonitorElement *mSumCaloVsPtInitial_0p522_0p783;
  MonitorElement *mSumCaloVsPtInitial_0p783_1p131;
  MonitorElement *mSumCaloVsPtInitial_1p131_1p479;
  MonitorElement *mSumCaloVsPtInitial_1p479_1p740;
  MonitorElement *mSumCaloVsPtInitial_1p740_2p043;
  MonitorElement *mSumCaloVsPtInitial_2p043_2p650;
  MonitorElement *mSumCaloVsPtInitial_2p650_5p191;

  MonitorElement *mSumCaloVsPt_n5p191_n2p650;
  MonitorElement *mSumCaloVsPt_n2p650_n2p043;
  MonitorElement *mSumCaloVsPt_n2p043_n1p740;
  MonitorElement *mSumCaloVsPt_n1p740_n1p479;
  MonitorElement *mSumCaloVsPt_n1p479_n1p131;
  MonitorElement *mSumCaloVsPt_n1p131_n0p783;
  MonitorElement *mSumCaloVsPt_n0p783_n0p522;
  MonitorElement *mSumCaloVsPt_n0p522_0p522;
  MonitorElement *mSumCaloVsPt_0p522_0p783;
  MonitorElement *mSumCaloVsPt_0p783_1p131;
  MonitorElement *mSumCaloVsPt_1p131_1p479;
  MonitorElement *mSumCaloVsPt_1p479_1p740;
  MonitorElement *mSumCaloVsPt_1p740_2p043;
  MonitorElement *mSumCaloVsPt_2p043_2p650;
  MonitorElement *mSumCaloVsPt_2p650_5p191;

  MonitorElement *mSumCaloPt_n5p191_n2p650;
  MonitorElement *mSumCaloPt_n2p650_n2p043;
  MonitorElement *mSumCaloPt_n2p043_n1p740;
  MonitorElement *mSumCaloPt_n1p740_n1p479;
  MonitorElement *mSumCaloPt_n1p479_n1p131;
  MonitorElement *mSumCaloPt_n1p131_n0p783;
  MonitorElement *mSumCaloPt_n0p783_n0p522;
  MonitorElement *mSumCaloPt_n0p522_0p522;
  MonitorElement *mSumCaloPt_0p522_0p783;
  MonitorElement *mSumCaloPt_0p783_1p131;
  MonitorElement *mSumCaloPt_1p131_1p479;
  MonitorElement *mSumCaloPt_1p479_1p740;
  MonitorElement *mSumCaloPt_1p740_2p043;
  MonitorElement *mSumCaloPt_2p043_2p650;
  MonitorElement *mSumCaloPt_2p650_5p191;

  // Jet parameters
  MonitorElement *mEta;
  MonitorElement *mPhi;
  MonitorElement *mPt;
  MonitorElement *mP;
  MonitorElement *mEnergy;
  MonitorElement *mMass;
  MonitorElement *mConstituents;
  MonitorElement *mJetArea;
  MonitorElement *mjetpileup;
  MonitorElement *mNJets;
  MonitorElement *mNJets_40;

  // histograms added on Oct 27th to study the gen-reco
  MonitorElement *mGenEta;
  MonitorElement *mGenPhi;
  MonitorElement *mGenPt;
  MonitorElement *mPtHat;

  MonitorElement *mPtRecoOverGen_B_20_30_Cent_0_10;
  MonitorElement *mPtRecoOverGen_E_20_30_Cent_0_10;
  MonitorElement *mPtRecoOverGen_F_20_30_Cent_0_10;
  MonitorElement *mPtRecoOverGen_B_30_50_Cent_0_10;
  MonitorElement *mPtRecoOverGen_E_30_50_Cent_0_10;
  MonitorElement *mPtRecoOverGen_F_30_50_Cent_0_10;
  MonitorElement *mPtRecoOverGen_B_50_80_Cent_0_10;
  MonitorElement *mPtRecoOverGen_E_50_80_Cent_0_10;
  MonitorElement *mPtRecoOverGen_F_50_80_Cent_0_10;
  MonitorElement *mPtRecoOverGen_B_80_120_Cent_0_10;
  MonitorElement *mPtRecoOverGen_E_80_120_Cent_0_10;
  MonitorElement *mPtRecoOverGen_F_80_120_Cent_0_10;
  MonitorElement *mPtRecoOverGen_B_120_180_Cent_0_10;
  MonitorElement *mPtRecoOverGen_E_120_180_Cent_0_10;
  MonitorElement *mPtRecoOverGen_F_120_180_Cent_0_10;
  MonitorElement *mPtRecoOverGen_B_180_300_Cent_0_10;
  MonitorElement *mPtRecoOverGen_E_180_300_Cent_0_10;
  MonitorElement *mPtRecoOverGen_F_180_300_Cent_0_10;
  MonitorElement *mPtRecoOverGen_B_300_Inf_Cent_0_10;
  MonitorElement *mPtRecoOverGen_E_300_Inf_Cent_0_10;
  MonitorElement *mPtRecoOverGen_F_300_Inf_Cent_0_10;

  MonitorElement *mPtRecoOverGen_B_20_30_Cent_10_30;
  MonitorElement *mPtRecoOverGen_E_20_30_Cent_10_30;
  MonitorElement *mPtRecoOverGen_F_20_30_Cent_10_30;
  MonitorElement *mPtRecoOverGen_B_30_50_Cent_10_30;
  MonitorElement *mPtRecoOverGen_E_30_50_Cent_10_30;
  MonitorElement *mPtRecoOverGen_F_30_50_Cent_10_30;
  MonitorElement *mPtRecoOverGen_B_50_80_Cent_10_30;
  MonitorElement *mPtRecoOverGen_E_50_80_Cent_10_30;
  MonitorElement *mPtRecoOverGen_F_50_80_Cent_10_30;
  MonitorElement *mPtRecoOverGen_B_80_120_Cent_10_30;
  MonitorElement *mPtRecoOverGen_E_80_120_Cent_10_30;
  MonitorElement *mPtRecoOverGen_F_80_120_Cent_10_30;
  MonitorElement *mPtRecoOverGen_B_120_180_Cent_10_30;
  MonitorElement *mPtRecoOverGen_E_120_180_Cent_10_30;
  MonitorElement *mPtRecoOverGen_F_120_180_Cent_10_30;
  MonitorElement *mPtRecoOverGen_B_180_300_Cent_10_30;
  MonitorElement *mPtRecoOverGen_E_180_300_Cent_10_30;
  MonitorElement *mPtRecoOverGen_F_180_300_Cent_10_30;
  MonitorElement *mPtRecoOverGen_B_300_Inf_Cent_10_30;
  MonitorElement *mPtRecoOverGen_E_300_Inf_Cent_10_30;
  MonitorElement *mPtRecoOverGen_F_300_Inf_Cent_10_30;

  MonitorElement *mPtRecoOverGen_B_20_30_Cent_30_50;
  MonitorElement *mPtRecoOverGen_E_20_30_Cent_30_50;
  MonitorElement *mPtRecoOverGen_F_20_30_Cent_30_50;
  MonitorElement *mPtRecoOverGen_B_30_50_Cent_30_50;
  MonitorElement *mPtRecoOverGen_E_30_50_Cent_30_50;
  MonitorElement *mPtRecoOverGen_F_30_50_Cent_30_50;
  MonitorElement *mPtRecoOverGen_B_50_80_Cent_30_50;
  MonitorElement *mPtRecoOverGen_E_50_80_Cent_30_50;
  MonitorElement *mPtRecoOverGen_F_50_80_Cent_30_50;
  MonitorElement *mPtRecoOverGen_B_80_120_Cent_30_50;
  MonitorElement *mPtRecoOverGen_E_80_120_Cent_30_50;
  MonitorElement *mPtRecoOverGen_F_80_120_Cent_30_50;
  MonitorElement *mPtRecoOverGen_B_120_180_Cent_30_50;
  MonitorElement *mPtRecoOverGen_E_120_180_Cent_30_50;
  MonitorElement *mPtRecoOverGen_F_120_180_Cent_30_50;
  MonitorElement *mPtRecoOverGen_B_180_300_Cent_30_50;
  MonitorElement *mPtRecoOverGen_E_180_300_Cent_30_50;
  MonitorElement *mPtRecoOverGen_F_180_300_Cent_30_50;
  MonitorElement *mPtRecoOverGen_B_300_Inf_Cent_30_50;
  MonitorElement *mPtRecoOverGen_E_300_Inf_Cent_30_50;
  MonitorElement *mPtRecoOverGen_F_300_Inf_Cent_30_50;

  MonitorElement *mPtRecoOverGen_B_20_30_Cent_50_80;
  MonitorElement *mPtRecoOverGen_E_20_30_Cent_50_80;
  MonitorElement *mPtRecoOverGen_F_20_30_Cent_50_80;
  MonitorElement *mPtRecoOverGen_B_30_50_Cent_50_80;
  MonitorElement *mPtRecoOverGen_E_30_50_Cent_50_80;
  MonitorElement *mPtRecoOverGen_F_30_50_Cent_50_80;
  MonitorElement *mPtRecoOverGen_B_50_80_Cent_50_80;
  MonitorElement *mPtRecoOverGen_E_50_80_Cent_50_80;
  MonitorElement *mPtRecoOverGen_F_50_80_Cent_50_80;
  MonitorElement *mPtRecoOverGen_B_80_120_Cent_50_80;
  MonitorElement *mPtRecoOverGen_E_80_120_Cent_50_80;
  MonitorElement *mPtRecoOverGen_F_80_120_Cent_50_80;
  MonitorElement *mPtRecoOverGen_B_120_180_Cent_50_80;
  MonitorElement *mPtRecoOverGen_E_120_180_Cent_50_80;
  MonitorElement *mPtRecoOverGen_F_120_180_Cent_50_80;
  MonitorElement *mPtRecoOverGen_B_180_300_Cent_50_80;
  MonitorElement *mPtRecoOverGen_E_180_300_Cent_50_80;
  MonitorElement *mPtRecoOverGen_F_180_300_Cent_50_80;
  MonitorElement *mPtRecoOverGen_B_300_Inf_Cent_50_80;
  MonitorElement *mPtRecoOverGen_E_300_Inf_Cent_50_80;
  MonitorElement *mPtRecoOverGen_F_300_Inf_Cent_50_80;

  // generation profiles
  MonitorElement *mPtRecoOverGen_GenPt_B_Cent_0_10;
  MonitorElement *mPtRecoOverGen_GenPt_E_Cent_0_10;
  MonitorElement *mPtRecoOverGen_GenPt_F_Cent_0_10;
  MonitorElement *mPtRecoOverGen_GenPt_B_Cent_10_30;
  MonitorElement *mPtRecoOverGen_GenPt_E_Cent_10_30;
  MonitorElement *mPtRecoOverGen_GenPt_F_Cent_10_30;
  MonitorElement *mPtRecoOverGen_GenPt_B_Cent_30_50;
  MonitorElement *mPtRecoOverGen_GenPt_E_Cent_30_50;
  MonitorElement *mPtRecoOverGen_GenPt_F_Cent_30_50;
  MonitorElement *mPtRecoOverGen_GenPt_B_Cent_50_80;
  MonitorElement *mPtRecoOverGen_GenPt_E_Cent_50_80;
  MonitorElement *mPtRecoOverGen_GenPt_F_Cent_50_80;

  MonitorElement *mPtRecoOverGen_GenEta_20_30_Cent_0_10;
  MonitorElement *mPtRecoOverGen_GenEta_30_50_Cent_0_10;
  MonitorElement *mPtRecoOverGen_GenEta_50_80_Cent_0_10;
  MonitorElement *mPtRecoOverGen_GenEta_80_120_Cent_0_10;
  MonitorElement *mPtRecoOverGen_GenEta_120_180_Cent_0_10;
  MonitorElement *mPtRecoOverGen_GenEta_180_300_Cent_0_10;
  MonitorElement *mPtRecoOverGen_GenEta_300_Inf_Cent_0_10;

  MonitorElement *mPtRecoOverGen_GenEta_20_30_Cent_10_30;
  MonitorElement *mPtRecoOverGen_GenEta_30_50_Cent_10_30;
  MonitorElement *mPtRecoOverGen_GenEta_50_80_Cent_10_30;
  MonitorElement *mPtRecoOverGen_GenEta_80_120_Cent_10_30;
  MonitorElement *mPtRecoOverGen_GenEta_120_180_Cent_10_30;
  MonitorElement *mPtRecoOverGen_GenEta_180_300_Cent_10_30;
  MonitorElement *mPtRecoOverGen_GenEta_300_Inf_Cent_10_30;

  MonitorElement *mPtRecoOverGen_GenEta_20_30_Cent_30_50;
  MonitorElement *mPtRecoOverGen_GenEta_30_50_Cent_30_50;
  MonitorElement *mPtRecoOverGen_GenEta_50_80_Cent_30_50;
  MonitorElement *mPtRecoOverGen_GenEta_80_120_Cent_30_50;
  MonitorElement *mPtRecoOverGen_GenEta_120_180_Cent_30_50;
  MonitorElement *mPtRecoOverGen_GenEta_180_300_Cent_30_50;
  MonitorElement *mPtRecoOverGen_GenEta_300_Inf_Cent_30_50;

  MonitorElement *mPtRecoOverGen_GenEta_20_30_Cent_50_80;
  MonitorElement *mPtRecoOverGen_GenEta_30_50_Cent_50_80;
  MonitorElement *mPtRecoOverGen_GenEta_50_80_Cent_50_80;
  MonitorElement *mPtRecoOverGen_GenEta_80_120_Cent_50_80;
  MonitorElement *mPtRecoOverGen_GenEta_120_180_Cent_50_80;
  MonitorElement *mPtRecoOverGen_GenEta_180_300_Cent_50_80;
  MonitorElement *mPtRecoOverGen_GenEta_300_Inf_Cent_50_80;

  MonitorElement *mPFCandpT_vs_eta_Unknown;        // pf id 0
  MonitorElement *mPFCandpT_vs_eta_ChargedHadron;  // pf id - 1
  MonitorElement *mPFCandpT_vs_eta_electron;       // pf id - 2
  MonitorElement *mPFCandpT_vs_eta_muon;           // pf id - 3
  MonitorElement *mPFCandpT_vs_eta_photon;         // pf id - 4
  MonitorElement *mPFCandpT_vs_eta_NeutralHadron;  // pf id - 5
  MonitorElement *mPFCandpT_vs_eta_HadE_inHF;      // pf id - 6
  MonitorElement *mPFCandpT_vs_eta_EME_inHF;       // pf id - 7

  MonitorElement *mPFCandpT_Barrel_Unknown;        // pf id 0
  MonitorElement *mPFCandpT_Barrel_ChargedHadron;  // pf id - 1
  MonitorElement *mPFCandpT_Barrel_electron;       // pf id - 2
  MonitorElement *mPFCandpT_Barrel_muon;           // pf id - 3
  MonitorElement *mPFCandpT_Barrel_photon;         // pf id - 4
  MonitorElement *mPFCandpT_Barrel_NeutralHadron;  // pf id - 5
  MonitorElement *mPFCandpT_Barrel_HadE_inHF;      // pf id - 6
  MonitorElement *mPFCandpT_Barrel_EME_inHF;       // pf id - 7

  MonitorElement *mPFCandpT_Endcap_Unknown;        // pf id 0
  MonitorElement *mPFCandpT_Endcap_ChargedHadron;  // pf id - 1
  MonitorElement *mPFCandpT_Endcap_electron;       // pf id - 2
  MonitorElement *mPFCandpT_Endcap_muon;           // pf id - 3
  MonitorElement *mPFCandpT_Endcap_photon;         // pf id - 4
  MonitorElement *mPFCandpT_Endcap_NeutralHadron;  // pf id - 5
  MonitorElement *mPFCandpT_Endcap_HadE_inHF;      // pf id - 6
  MonitorElement *mPFCandpT_Endcap_EME_inHF;       // pf id - 7

  MonitorElement *mPFCandpT_Forward_Unknown;        // pf id 0
  MonitorElement *mPFCandpT_Forward_ChargedHadron;  // pf id - 1
  MonitorElement *mPFCandpT_Forward_electron;       // pf id - 2
  MonitorElement *mPFCandpT_Forward_muon;           // pf id - 3
  MonitorElement *mPFCandpT_Forward_photon;         // pf id - 4
  MonitorElement *mPFCandpT_Forward_NeutralHadron;  // pf id - 5
  MonitorElement *mPFCandpT_Forward_HadE_inHF;      // pf id - 6
  MonitorElement *mPFCandpT_Forward_EME_inHF;       // pf id - 7

  // Parameters

  bool isCaloJet;
  bool isJPTJet;
  bool isPFJet;

  static const Int_t fourierOrder_ = 5;
  static const Int_t etaBins_ = 15;

  static const size_t nedge_pseudorapidity = etaBins_ + 1;
};

#endif
