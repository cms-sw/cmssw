#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "AnalysisDataFormats/TopObjects/interface/StGenEvent.h"

class StGenEventReco : public edm::global::EDProducer<> {
public:
  explicit StGenEventReco(const edm::ParameterSet&);
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  edm::EDGetTokenT<reco::GenParticleCollection> srcToken_;
  edm::EDGetTokenT<reco::GenParticleCollection> initToken_;
  edm::EDPutTokenT<StGenEvent> putToken_;
};

StGenEventReco::StGenEventReco(const edm::ParameterSet& cfg)
    : srcToken_(consumes<reco::GenParticleCollection>(cfg.getParameter<edm::InputTag>("src"))),
      initToken_(consumes<reco::GenParticleCollection>(cfg.getParameter<edm::InputTag>("init"))),
      putToken_(produces<StGenEvent>()) {}

void StGenEventReco::produce(edm::StreamID, edm::Event& evt, const edm::EventSetup& setup) const {
  //add TopDecayTree
  reco::GenParticleRefProd cands(evt.getHandle(srcToken_));

  //add InitialStatePartons
  reco::GenParticleRefProd initParts(evt.getHandle(initToken_));

  //add genEvt to the output stream
  evt.emplace(putToken_, cands, initParts);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(StGenEventReco);
