//
// Author:  Jan Heyninck, Steven Lowette
// Created: Tue Apr  10 12:01:49 CEST 2007
//
// $Id: TopMuonProducer.h,v 1.7 2007/08/27 11:04:31 tsirig Exp $
//

#ifndef TopObjectProducers_TopMuonProducer_h
#define TopObjectProducers_TopMuonProducer_h

#include <string>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/Utilities/interface/PtComparator.h"
#include "AnalysisDataFormats/TopObjects/interface/TopLepton.h"

class TopObjectResolutionCalc;
class TopLeptonTrackerIsolationPt;
class TopLeptonCaloIsolationEnergy;
class TopLeptonLRCalc;


class TopMuonProducer : public edm::EDProducer {
  
 public:
  
  explicit TopMuonProducer(const edm::ParameterSet & iConfig);
  ~TopMuonProducer();
  virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);
  
 private:
  void matchTruth(const reco::CandidateCollection&, TopMuonTypeCollection&);  
  reco::GenParticleCandidate findTruth(const reco::CandidateCollection&, const TopMuonType&);  
  
 private:
  
  // configurables
  edm::InputTag muonSrc_, genPartSrc_, tracksTag_;
  bool useTrkIso_, useCalIso_; 
  bool addResolutions_, useNNReso_;  
  bool addLRValues_;
  bool doGenMatch_;
  std::string muonResoFile_;
  std::string muonLRFile_;
  double minRecoOnGenEt_, maxRecoOnGenEt_, maxDeltaR_;  

  // tools
  TopObjectResolutionCalc *theResoCalc_;
  TopLeptonTrackerIsolationPt  *trkIsolation_;
  TopLeptonCaloIsolationEnergy *calIsolation_;
  TopLeptonLRCalc *theLeptonLRCalc_;
  std::vector<std::pair<const reco::Candidate *, TopMuonType*> > pairGenRecoMuonsVector_;
  GreaterByPt<TopMuon> pTComparator_;
};


#endif
