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

class StDecaySubset : public edm::EDProducer {
 public:
  explicit StDecaySubset(const edm::ParameterSet&);
  ~StDecaySubset();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);
  void fillOutput(const reco::CandidateCollection&, reco::CandidateCollection&);
  void fillRefs(const reco::CandidateRefProd&, reco::CandidateCollection&);

  reco::Particle::LorentzVector fourVector(const reco::Candidate&);
 private:
  edm::InputTag src_;
  int switchOption;
  std::map<int,std::vector<int> > refs_; //management of daughter
                                         //indices for fillRefs
};
