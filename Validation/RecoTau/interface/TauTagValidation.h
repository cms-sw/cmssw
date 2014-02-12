#ifndef TauTagValidation_h
#define TauTagValidation_h

// -*- C++ -*-
//
// Package:    TauTagValidation
// Class:      TauTagValidation
//
/* *\class TauTagValidation TauTagValidation.cc

 Description: EDAnalyzer to validate the Collections from the ConeIsolation Producer
 It is supposed to be used for Offline Tau Reconstrction, so PrimaryVertex should be used.
 Implementation:

*/
// Original Author: Ricardo Vasquez Sierra On August 29, 2007
// user include files


#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/TauReco/interface/CaloTauDiscriminator.h"

#include "RecoParticleFlow/Benchmark/interface/PFBenchmarkAlgo.h"

// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"
#include "TLorentzVector.h"
#include "TH1D.h"
#include "TH1.h"
#include "TH1F.h"
#include <vector>
#include <string>

// Include DQM core
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

typedef math::XYZTLorentzVectorD  LV;
typedef std::vector<LV>  LVCollection;

struct hinfo{
  int nbins;
  double min;
  double max;
  hinfo(int n, double m, double M){
    nbins = n;
    min = m;
    max = M;
  }
  hinfo(const edm::ParameterSet& config){
    nbins = config.getParameter<int>("nbins");
    min = config.getParameter<double>("min");
    max = config.getParameter<double>("max");
  }
};

// class declaration
class TauTagValidation : public edm::EDAnalyzer {

public:
  explicit TauTagValidation(const edm::ParameterSet&);
  ~TauTagValidation();

  virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  virtual void beginJob();
  virtual void endJob();
  virtual void beginRun(edm::Run const&, edm::EventSetup const&);
  virtual void endRun(edm::Run const&, edm::EventSetup const&);
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

private:
  /// label of the current module
  std::string moduleLabel_;
  ///sum the transversal momentum of all candidates
  double getSumPt(const std::vector<edm::Ptr<reco::PFCandidate> > & candidates);
  ///get rid of redundant parts to shorten the label
  bool stripDiscriminatorLabel(const std::string& discriminatorLabel, std::string & newLabel);

  edm::ParameterSet histoSettings_;
  /// generic access to dynamic trigger table
  GenericTriggerEventFlag * genericTriggerEventFlag_;
    
  /// What's the reference for the Validation Leptons or Jets
  std::string dataType_;

 // Matching criteria
  double matchDeltaR_Leptons_;
  double matchDeltaR_Jets_;
  double TauPtCut_;
  
  //optional: filter candidates by passed cuts
  std::string recoCuts_, genCuts_;

  // output histograms
  bool saveoutputhistograms_, turnOnTrigger_;

 // Reference Collection
  edm::InputTag refCollectionInputTag_;
  std::string refCollection_;

  // In case you need to distinguish the output file
  std::string extensionName_;

  // Reconstructed product of interest
  edm::InputTag TauProducerInputTag_, PrimaryVertexCollection_;
  std::string TauProducer_;

  // std::vector<std::string> TauProducerDiscriminators_;
  // std::vector<double> TauDiscriminatorCuts_;

  std::vector< edm::ParameterSet > discriminators_;

  // CMSSW version

  std::string tversion;
  std::string outPutFile_;

  std::map<std::string,  MonitorElement *> ptTauVisibleMap;
  std::map<std::string,  MonitorElement *> etaTauVisibleMap;
  std::map<std::string,  MonitorElement *> phiTauVisibleMap;
  std::map<std::string,  MonitorElement *> pileupTauVisibleMap;
  std::map<std::string,  MonitorElement *> plotMap_;

  // All the extra MonitorElements that we would like to add for each Tau Tagging step
  // First for the PFTaus
  // Number of PFTau Candidates with a leading charged hadron in it (within a cone of 0.1 avound the jet axis and a minimum pt of 6 GeV)

  MonitorElement* nPFJet_LeadingChargedHadron_ChargedHadronsSignal_;
  MonitorElement* nPFJet_LeadingChargedHadron_ChargedHadronsIsolAnnulus_;
  MonitorElement* nPFJet_LeadingChargedHadron_GammasSignal_;
  MonitorElement* nPFJet_LeadingChargedHadron_GammasIsolAnnulus_;
  MonitorElement* nPFJet_LeadingChargedHadron_NeutralHadronsSignal_;
  MonitorElement* nPFJet_LeadingChargedHadron_NeutralHadronsIsolAnnulus_;

  // Isolated PFTau with a Leading charged hadron with no Charged Hadrons inside the isolation annulus

  MonitorElement* nIsolated_NoChargedHadrons_ChargedHadronsSignal_;
  MonitorElement* nIsolated_NoChargedHadrons_GammasSignal_;
  MonitorElement* nIsolated_NoChargedHadrons_GammasIsolAnnulus_;
  MonitorElement* nIsolated_NoChargedHadrons_NeutralHadronsSignal_;
  MonitorElement* nIsolated_NoChargedHadrons_NeutralHadronsIsolAnnulus_;

  // Isolated PFTau with a Leading charge hadron with no Charged Hadron inside the isolation annulus with no Ecal/Gamma candidates in the isolation annulus

  MonitorElement* nIsolated_NoChargedNoGammas_ChargedHadronsSignal_;
  MonitorElement* nIsolated_NoChargedNoGammas_GammasSignal_;
  MonitorElement* nIsolated_NoChargedNoGammas_NeutralHadronsSignal_;
  MonitorElement* nIsolated_NoChargedNoGammas_NeutralHadronsIsolAnnulus_;


  // Second for the CaloTaus
  // Number of CaloJets with a Leading Track (within a cone of 0.1 around the jet axis and a minimum pt of 5. GeV)

  MonitorElement* nCaloJet_LeadingTrack_signalTracksInvariantMass_;
  MonitorElement* nCaloJet_LeadingTrack_signalTracks_;
  MonitorElement* nCaloJet_LeadingTrack_isolationTracks_;
  MonitorElement* nCaloJet_LeadingTrack_isolationECALhitsEtSum_;

 // Track Isolated CaloTau with a Leading Track

  MonitorElement* nTrackIsolated_isolationECALhitsEtSum_;
  MonitorElement* nTrackIsolated_signalTracksInvariantMass_;
  MonitorElement* nTrackIsolated_signalTracks_;

 // EM Isolated CaloTau with a Leading with no tracks in the Isolation Annulus

  MonitorElement* nEMIsolated_signalTracksInvariantMass_;
  MonitorElement* nEMIsolated_signalTracks_;

  // book-keeping variables

  DQMStore* dbeTau_;
  int numEvents_;

 protected:

  PFBenchmarkAlgo *algo_;

 private:
  bool chainCuts_;

};

#endif
