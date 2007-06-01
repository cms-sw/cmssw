#include <memory>
#include <string>
#include <vector>
#include <map>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"

class TtGenEventReco : public edm::EDProducer {
 public:
  explicit TtGenEventReco(const edm::ParameterSet&);
  ~TtGenEventReco();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);
  void fillOutput(const reco::CandidateCollection&, reco::CandidateCollection&);
  void fillRefs(const reco::CandidateRefProd&, reco::CandidateCollection&);
 private:
  edm::InputTag src_;  
  std::map<int,std::vector<int> > refs_;
};
