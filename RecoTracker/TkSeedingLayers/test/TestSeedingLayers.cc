#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"

#include <string>
#include <iostream>

struct TestSeedingLayers final : public edm::global::EDAnalyzer<> {
  explicit TestSeedingLayers(const edm::ParameterSet& conf)
      : SeedingLayerSetsHits_token(consumes<SeedingLayerSetsHits>(edm::InputTag("MixedLayerTriplets"))) {}
  virtual ~TestSeedingLayers() = default;

  void analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const&) const {
    edm::Handle<SeedingLayerSetsHits> layersH = iEvent.getHandle(SeedingLayerSetsHits_token);
    auto const& layers = *layersH;

    edm::LogPrint("TestSeedingLayers") << layers.numberOfLayersInSet() << ' ' << layers.size();

    for (auto const& lset : layers) {
      edm::LogPrint("TestSeedingLayers") << lset.size();
      for (auto const& la : lset) {
        edm::LogPrint("TestSeedingLayers") << ": " << la.name() << ' ' << la.index() << ' ' << la.hits().size();
      }
      edm::LogPrint("TestSeedingLayers");
    }
  }

private:
  edm::EDGetTokenT<SeedingLayerSetsHits> SeedingLayerSetsHits_token;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TestSeedingLayers);
