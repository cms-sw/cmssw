#ifndef TtSemiHypothesisMVADisc_h
#define TtSemiHypothesisMVADisc_h

#include <memory>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

#include "DataFormats/Candidate/interface/CandidateWithRef.h"
#include "DataFormats/Candidate/interface/ShallowCloneCandidate.h"
#include "DataFormats/Candidate/interface/NamedCompositeCandidate.h"


class TtSemiHypothesisMVADisc : public edm::EDProducer {

 public:

  explicit TtSemiHypothesisMVADisc(const edm::ParameterSet&);
  ~TtSemiHypothesisMVADisc();

 private:
  
  virtual void produce(edm::Event&, const edm::EventSetup&);
  bool isValid(const int& idx, const edm::Handle<std::vector<pat::Jet> >& jets){ return (0<=idx && idx<(int)jets->size()); };
  reco::NamedCompositeCandidate buildHypo(const edm::Handle<std::vector<pat::Jet> >&,
					  const edm::Handle<edm::View<reco::RecoCandidate> >&,
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
