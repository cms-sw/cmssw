#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
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

TtGenEventReco::TtGenEventReco(const edm::ParameterSet& cfg)
    : srcToken_(consumes<reco::GenParticleCollection>(cfg.getParameter<edm::InputTag>("src"))),
      initToken_(consumes<reco::GenParticleCollection>(cfg.getParameter<edm::InputTag>("init"))) {
  produces<TtGenEvent>();
}

TtGenEventReco::~TtGenEventReco() {}

void TtGenEventReco::produce(edm::Event& evt, const edm::EventSetup& setup) {
  edm::Handle<reco::GenParticleCollection> parts;
  evt.getByToken(srcToken_, parts);

  edm::Handle<reco::GenParticleCollection> inits;
  evt.getByToken(initToken_, inits);

  //add TopDecayTree
  reco::GenParticleRefProd cands(parts);

  //add InitialStatePartons
  reco::GenParticleRefProd initParts(inits);

  //add genEvt to the output stream
  TtGenEvent* genEvt = new TtGenEvent(cands, initParts);
  std::unique_ptr<TtGenEvent> gen(genEvt);
  evt.put(std::move(gen));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TtGenEventReco);
