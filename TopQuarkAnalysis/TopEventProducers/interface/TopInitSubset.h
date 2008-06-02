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
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

namespace TopInitID{
  static const int status = 3;
  static const int tID    = 6; 
}

class TopInitSubset : public edm::EDProducer {

 public:

  explicit TopInitSubset(const edm::ParameterSet&);
  ~TopInitSubset();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);
  void fillOutput(const reco::GenParticleCollection&, reco::GenParticleCollection&);

 private:

  edm::InputTag src_;  
};
