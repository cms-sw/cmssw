#ifndef TauTagVal_h
#define TauTagVal_h

// -*- C++ -*-
//
// Package:    TauTagVal
// Class:      TauTagVal
// 
/**\class TauTagVal TauTagVal.cc 

 Description: EDAnalyzer to validate the Collections from the ConeIsolation Producer
 It is supposed to be used for Offline Tau Reconstrction, so PrimaryVertex should be used.
 Implementation:

*/
// Original Author:  Simone Gennai
// Modified by Ricardo Vasquez Sierra On August 29, 2007
// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "HepMC/GenParticle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/BTauReco/interface/EMIsolatedTauTagInfo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "DataFormats/JetReco/interface/PFJet.h"
//#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/JetReco/interface/GenJet.h"

// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"
#include "TLorentzVector.h"
#include "TH1D.h"
#include "TH1.h"
#include "TH1F.h"
#include <vector>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


// class declaration    
class TauTagVal : public edm::EDAnalyzer {

public:
  explicit TauTagVal(const edm::ParameterSet&);
  ~TauTagVal() {}

  virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  virtual void beginJob();
  virtual void endJob();

  //  Function to get the Daughters of a Generated Particle 

private:
  //------------HELPER FUNCTIONS---------------------------


  std::vector<TLorentzVector> getVectorOfVisibleTauJets(HepMC::GenEvent *theEvent);
  std::vector<HepMC::GenParticle*> getGenStableDecayProducts(const HepMC::GenParticle* particle);
  std::vector<TLorentzVector> getVectorOfGenJets(edm::Handle< reco::GenJetCollection >& genJets );

  // ----------- MEMBER DATA--------------------------------
  enum tauDecayModes {kElectron, kMuon, 
		      kOneProng0pi0, kOneProng1pi0, kOneProng2pi0,
		      kThreeProng0pi0, kThreeProng1pi0,
		      kOther, kUndefined};

  edm::InputTag jetTagSrc_, jetEMTagSrc_, genJetSrc_;
  
  std::string outPutFile_;
  float rSig_,rMatch_,ptLeadTk_, rIso_, minPtIsoRing_;
  int nTracksInIsolationRing_;
  std::string dataType_;
  std::string outputhistograms_; 
  std::string tversion;
  //AGGIUNGERE MC INFO???

  // MonteCarlo Taus -- to see what kind of Taus do we originally have!
  MonitorElement* ptTauMC_;
  MonitorElement* etaTauMC_;
  MonitorElement* phiTauMC_;
  MonitorElement* energyTauMC_;
  MonitorElement* hGenTauDecay_DecayModes_;
  MonitorElement* hGenTauDecay_DecayModesChosen_;
  MonitorElement* nMCTaus_ptTauJet_;
  MonitorElement* nMCTaus_etaTauJet_;
  MonitorElement* nMCTaus_phiTauJet_;
  MonitorElement* nMCTaus_energyTauJet_;

 
  // Leading Track Related Histograms In case the finding of the leading track is a problem
  MonitorElement* deltaRLeadTk_Jet_;
  MonitorElement* ptLeadingTrack_;

  // The following histograms count the number of matched Montecarlo to isolatedTauTagInfoCollection
  MonitorElement* nRecoJet_ptTauJet_;
  MonitorElement* nRecoJet_etaTauJet_;
  MonitorElement* nRecoJet_phiTauJet_;
  MonitorElement* nRecoJet_energyTauJet_;
  MonitorElement* nAssociatedTracks_;   // for recoJets
  MonitorElement* nSelectedTracks_;     // for recoJets

  // The following histograms count  of RecoJets that are matched to MC Tau with a LeadingTrack of 6.0 GeV
  MonitorElement* nRecoJet_LeadingTrack_ptTauJet_;
  MonitorElement* nRecoJet_LeadingTrack_etaTauJet_;
  MonitorElement* nRecoJet_LeadingTrack_phiTauJet_;
  MonitorElement* nRecoJet_LeadingTrack_energyTauJet_;  
  MonitorElement* nSignalTracks_;                     // Signal Tracks in IsolatedTauTagInfo after finding leading track in rMatch=0.1 and pt 1.0

  // The following histograms count the number of isolated isolatedTauTagInfoCollection
  MonitorElement* nIsolatedJet_ptTauJet_;
  MonitorElement* nIsolatedJet_etaTauJet_;
  MonitorElement* nIsolatedJet_phiTauJet_;
  MonitorElement* nIsolatedJet_energyTauJet_;
  MonitorElement* nSignalTracksAfterIsolation_;       // Same as above but after Isolation
  MonitorElement* nIsolatedTausLeadingTrackPt_;
  MonitorElement* nIsolatedTausDeltaR_LTandJet_;
  MonitorElement* nAssociatedTracks_of_IsolatedTaus_;
  MonitorElement* nSelectedTracks_of_IsolatedTaus_;


// The following histograms count the number of EM isolated isolatedTauTagInfoCollection
  MonitorElement* nEMIsolatedJet_ptTauJet_;
  MonitorElement* nEMIsolatedJet_etaTauJet_;
  MonitorElement* nEMIsolatedJet_phiTauJet_;
  MonitorElement* nEMIsolatedJet_energyTauJet_;


  // What is the behaviour of cone isolation size on tagging of MC Taus (CONE_MATCHING_CRITERIA) 
  MonitorElement* nTausTotvsConeIsolation_;
  MonitorElement* nTausTaggedvsConeIsolation_;
  MonitorElement* nTausTotvsConeSignal_;
  MonitorElement* nTausTaggedvsConeSignal_;
  MonitorElement* nTausTotvsPtLeadingTrack_;
  MonitorElement* nTausTaggedvsPtLeadingTrack_;
  MonitorElement* nTausTotvsMatchingConeSize_;
  MonitorElement* nTausTaggedvsMatchingConeSize_;
 
  // book-keeping variables
 
  int numEvents_;

};

#endif











