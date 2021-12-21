#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "AnalysisDataFormats/TopObjects/interface/StGenEvent.h"

class StGenEventReco : public edm::EDProducer {
public:
  explicit StGenEventReco(const edm::ParameterSet&);
  ~StGenEventReco() override;
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<reco::GenParticleCollection> srcToken_;
  edm::EDGetTokenT<reco::GenParticleCollection> initToken_;
};

StGenEventReco::StGenEventReco(const edm::ParameterSet& cfg)
    : srcToken_(consumes<reco::GenParticleCollection>(cfg.getParameter<edm::InputTag>("src"))),
      initToken_(consumes<reco::GenParticleCollection>(cfg.getParameter<edm::InputTag>("init"))) {
  produces<StGenEvent>();
}

StGenEventReco::~StGenEventReco() {}

void StGenEventReco::produce(edm::Event& evt, const edm::EventSetup& setup) {
  edm::Handle<reco::GenParticleCollection> parts;
  evt.getByToken(srcToken_, parts);

  edm::Handle<reco::GenParticleCollection> inits;
  evt.getByToken(initToken_, inits);

  //add TopDecayTree
  reco::GenParticleRefProd cands(parts);

  //add InitialStatePartons
  reco::GenParticleRefProd initParts(inits);

  //add genEvt to the output stream
  StGenEvent* genEvt = new StGenEvent(cands, initParts);
  std::unique_ptr<StGenEvent> gen(genEvt);
  evt.put(std::move(gen));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(StGenEventReco);
