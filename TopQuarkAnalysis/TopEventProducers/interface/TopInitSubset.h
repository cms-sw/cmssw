#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

namespace TopInitID{
  static const int status = 3;
  static const int tID    = 6;
}

class TopInitSubset : public edm::EDProducer {

 public:

  explicit TopInitSubset(const edm::ParameterSet&);
  ~TopInitSubset() override;

  void produce(edm::Event&, const edm::EventSetup&) override;
  void fillOutput(const reco::GenParticleCollection&, reco::GenParticleCollection&);

 private:

  edm::EDGetTokenT<reco::GenParticleCollection> srcToken_;
};
