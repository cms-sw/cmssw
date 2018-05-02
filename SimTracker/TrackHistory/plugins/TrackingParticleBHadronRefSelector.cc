#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Common/interface/EDProductfwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"

#include "SimTracker/TrackHistory/interface/HistoryBase.h"

#include "HepPDT/ParticleID.hh"

class TrackingParticleBHadronRefSelector: public edm::stream::EDProducer<> {
public:
  TrackingParticleBHadronRefSelector(const edm::ParameterSet& iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  edm::EDGetTokenT<TrackingParticleCollection> tpToken_;

  HistoryBase tracer_;
};


TrackingParticleBHadronRefSelector::TrackingParticleBHadronRefSelector(const edm::ParameterSet& iConfig):
  tpToken_(consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>("src")))
{
  tracer_.depth(-2); // as in SimTracker/TrackHistory/src/TrackClassifier.cc

  produces<TrackingParticleRefVector>();
}

void TrackingParticleBHadronRefSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("mix", "MergedTrackTruth"));
  descriptions.add("trackingParticleBHadronRefSelectorDefault", desc);
}

void TrackingParticleBHadronRefSelector::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<TrackingParticleCollection> h_tps;
  iEvent.getByToken(tpToken_, h_tps);

  auto ret = std::make_unique<TrackingParticleRefVector>();

  // Logic is similar to SimTracker/TrackHistory
  for(size_t i=0, end=h_tps->size(); i<end; ++i) {
    auto tpRef = TrackingParticleRef(h_tps, i);
    if(tracer_.evaluate(tpRef)) { // ignore TP if history can not be traced
      // following is from TrackClassifier::processesAtGenerator()
      HistoryBase::RecoGenParticleTrail const & recoGenParticleTrail = tracer_.recoGenParticleTrail();
      for(const auto& particle: recoGenParticleTrail) {
        HepPDT::ParticleID particleID(particle->pdgId());
        if(particleID.hasBottom()) {
          ret->push_back(tpRef);
          break;
        }
      }
    }
  }

  iEvent.put(std::move(ret));
}

DEFINE_FWK_MODULE(TrackingParticleBHadronRefSelector);
