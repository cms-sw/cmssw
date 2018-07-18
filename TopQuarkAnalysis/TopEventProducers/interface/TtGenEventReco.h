#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

class TtGenEventReco : public edm::EDProducer {

 public:

  explicit TtGenEventReco(const edm::ParameterSet&);
  ~TtGenEventReco() override;
  void produce(edm::Event&, const edm::EventSetup&) override;

 private:

  edm::EDGetTokenT<reco::GenParticleCollection> srcToken_;
  edm::EDGetTokenT<reco::GenParticleCollection> initToken_;
};
