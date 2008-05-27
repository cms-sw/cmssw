#ifndef TtSemiGenMatchHypothesis_h
#define TtSemiGenMatchHypothesis_h

#include <memory>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

#include "DataFormats/Candidate/interface/CandidateWithRef.h"
#include "DataFormats/Candidate/interface/ShallowCloneCandidate.h"
#include "DataFormats/Candidate/interface/NamedCompositeCandidate.h"


class TtSemiGenMatchHypothesis : public edm::EDProducer {

 public:

  explicit TtSemiGenMatchHypothesis(const edm::ParameterSet&);
  ~TtSemiGenMatchHypothesis();

 private:
  
  virtual void produce(edm::Event&, const edm::EventSetup&);
  bool isValid(const int& idx, const edm::Handle<std::vector<pat::Jet> >& jets){ return (0<=idx && idx<(int)jets->size()); };
  reco::NamedCompositeCandidate buildHypo(const edm::Handle<std::vector<pat::Jet> >&,
					  const edm::Handle<std::vector<pat::Muon> >&,
					  const edm::Handle<std::vector<pat::MET> >&,
					  const edm::Handle<std::vector<int> >&);
  reco::NamedCompositeCandidate fillHypo (std::vector<reco::ShallowCloneCandidate>&);

 private:

  edm::InputTag jets_;
  edm::InputTag leps_;
  edm::InputTag mets_;

  edm::InputTag match_;  
};

#endif
