#ifndef ValidationRecoJetsJetTester_HeavyIons_h
#define ValidationRecoJetsJetTester_HeavyIons_h

// Producer for validation histograms for Calo, JPT and PF jet objects
// F. Ratnikov, Sept. 7, 2006
// Modified by Chiyoung Jeong, Feb. 2, 2010
// Modified by J. Piedra, Sept. 11, 2013
// Rewritten by Viola Sordini, Matthias Artur Weber, Robert Schoefbeck Nov./Dez. 2013
// Modified by Raghav Kunnawalkam Elayavalli, Aug 18th 2014 to run in 72X 
//                                          , Oct 22nd 2014 to run in 73X
//                                          , Dec 10th 2014 74X and adding the PF candidates information to easily detect  
//                                                          the voronoi subtraction algorithm failure modes.  

#include <cmath>
#include <string>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/JPTJet.h"
#include "DataFormats/JetReco/interface/JPTJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"

// include the pf candidates 
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
// include the voronoi subtraction
#include "DataFormats/HeavyIonEvent/interface/VoronoiBackground.h"
#include "RecoHI/HiJetAlgos/interface/UEParameters.h"
// include the centrality variables
#include "RecoHI/HiCentralityAlgos/interface/CentralityProvider.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "RecoJets/JetProducers/interface/JetMatchingTools.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

const Int_t MAXPARTICLE = 10000;

class MonitorElement;

class JetTester_HeavyIons : public DQMEDAnalyzer {
 public:

  explicit JetTester_HeavyIons (const edm::ParameterSet&);
  virtual ~JetTester_HeavyIons();

  virtual void analyze(const edm::Event&, const edm::EventSetup&); 
  virtual void beginJob();
  virtual void endJob();
  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) ;

 private:
  
  void fillMatchHists(const double GenEta,  const double GenPhi,  const double GenPt,
		      const double RecoEta, const double RecoPhi, const double RecoPt);
  
  edm::InputTag   mInputCollection;
  edm::InputTag   mInputGenCollection;
  edm::InputTag   mInputPFCandCollection;
//  edm::InputTag   rhoTag;
  edm::InputTag   centrality;
  
  std::string     mOutputFile;
  std::string     JetType;
  std::string     UEAlgo;
  edm::InputTag   Background;
  double          mRecoJetPtThreshold;
  double          mMatchGenPtThreshold;
  double          mGenEnergyFractionThreshold;
  double          mReverseEnergyFractionThreshold;
  double          mRThreshold;
  std::string     JetCorrectionService;

  //Tokens
  edm::EDGetTokenT<std::vector<reco::Vertex> > pvToken_;
  edm::EDGetTokenT<CaloTowerCollection > caloTowersToken_;
  edm::EDGetTokenT<reco::CaloJetCollection> caloJetsToken_;
  edm::EDGetTokenT<reco::PFJetCollection> pfJetsToken_;
  edm::EDGetTokenT<reco::BasicJetCollection> basicJetsToken_;
  edm::EDGetTokenT<reco::JPTJetCollection> jptJetsToken_;
  edm::EDGetTokenT<reco::GenJetCollection> genJetsToken_;
  edm::EDGetTokenT<edm::HepMCProduct> evtToken_;
  edm::EDGetTokenT<reco::PFCandidateCollection> pfCandToken_; 
  edm::EDGetTokenT<reco::CandidateView> pfCandViewToken_;
  //edm::EDGetTokenT<reco::VoronoiMap> backgrounds_;
  edm::EDGetTokenT<edm::ValueMap<reco::VoronoiBackground>> backgrounds_;
  edm::EDGetTokenT<std::vector<float>> backgrounds_value_;
  edm::EDGetTokenT<reco::Centrality> centralityToken_;
  
  //Include Particle flow variables 
  MonitorElement *mNPFpart;
  MonitorElement *mPFPt;
  MonitorElement *mPFEta;
  MonitorElement *mPFPhi;
  MonitorElement *mPFVsPt;
  MonitorElement *mPFVsPtInitial;
  MonitorElement *mPFVsPtEqualized;
  MonitorElement *mPFArea;
  MonitorElement *mSumpt;
  MonitorElement *mvn;
  MonitorElement *mpsin;
  // MonitorElement *ueraw;  

  // necessary plots for the vs validation which is the vn weighted SumpT for differnet eta bins, 
  MonitorElement *mSumPFVsPt;
  MonitorElement *mSumPFVsPtInitial;
  MonitorElement *mSumPFPt;

  MonitorElement *mSumPFVsPtInitial_eta;
  MonitorElement *mSumPFVsPt_eta;
  MonitorElement *mSumPFPt_eta;

  // Event variables (including centrality)
  MonitorElement* mNvtx;
  MonitorElement* mHF;

  // new additions Jan 12th 2015
  MonitorElement *mSumPFVsPt_HF;
  MonitorElement *mSumPFVsPtInitial_HF;
  MonitorElement *mSumPFPt_HF;
  MonitorElement *mPFVsPtInitial_eta_phi;
  MonitorElement *mPFVsPt_eta_phi;
  MonitorElement *mPFPt_eta_phi;
  //MonitorElement *mSumDeltapT_HF;
  MonitorElement *mDeltapT;
  MonitorElement *mDeltapT_eta;
  //MonitorElement *mDeltapT_phiMinusPsi2;
  MonitorElement *mDeltapT_eta_phi;
  //MonitorElement *mS
  
  // Jet parameters
  MonitorElement* mEta;
  MonitorElement* mPhi;
  MonitorElement* mPt;
  MonitorElement* mP;
  MonitorElement* mEnergy;
  MonitorElement* mMass;
  MonitorElement* mConstituents;
  MonitorElement* mJetArea;
  MonitorElement* mjetpileup;
  MonitorElement* mNJets_40;

  // Parameters

  bool            isCaloJet;
  bool            isJPTJet;
  bool            isPFJet;

  static const Int_t fourierOrder_ = 5;
  static const Int_t etaBins_ = 15;

  static const size_t nedge_pseudorapidity = etaBins_ + 1;

};

#endif
