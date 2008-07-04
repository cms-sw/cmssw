#ifndef TtSemiHypothesis_h
#define TtSemiHypothesis_h

#include <memory>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

#include "DataFormats/Candidate/interface/ShallowCloneCandidate.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvent.h"


class TtSemiHypothesis : public edm::EDProducer {

 public:

  explicit TtSemiHypothesis(const edm::ParameterSet&);
  ~TtSemiHypothesis();

 protected:
  
  /// produce the event hypothesis as CompositeCandidate and Key
  virtual void produce(edm::Event&, const edm::EventSetup&);
  /// return key
  int key() const { return key_; };
  /// return event hypothesis
  reco::CompositeCandidate hypo();
  /// check if index is in valid range of selected jets
  bool isValid(const int& idx, const edm::Handle<std::vector<pat::Jet> >& jets){ return (0<=idx && idx<(int)jets->size()); };

  // -----------------------------------------
  // implemet the following two functions
  // for a concrete event hypothesis
  // -----------------------------------------

  /// build the event hypothesis key
  virtual void buildKey() = 0;
  /// build event hypothesis from the reco objects of a semi-leptonic event 
  virtual void buildHypo(const edm::Handle<edm::View<reco::RecoCandidate> >& lepton, 
			 const edm::Handle<std::vector<pat::MET> >& neutrino, 
			 const edm::Handle<std::vector<pat::Jet> >& jets, 
			 const edm::Handle<std::vector<int> >& jetPartonAssociation) = 0;

 protected:

  edm::InputTag jets_;
  edm::InputTag leps_;
  edm::InputTag mets_;
  edm::InputTag match_;  

  int key_;

  reco::ShallowClonePtrCandidate *lightQ_;
  reco::ShallowClonePtrCandidate *lightQBar_;
  reco::ShallowClonePtrCandidate *hadronicB_;
  reco::ShallowClonePtrCandidate *leptonicB_;
  reco::ShallowClonePtrCandidate *neutrino_;
  reco::ShallowClonePtrCandidate *lepton_;
};

#endif
