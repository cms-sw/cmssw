#ifndef RecoTracker_TkTrackingRegions_TrackingRegionEDProducerT_H
#define RecoTracker_TkTrackingRegions_TrackingRegionEDProducerT_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"

#include <memory>
#include <vector>

template <typename T_TrackingRegionProducer>
class TrackingRegionEDProducerT : public edm::stream::EDProducer<> {
public:
  TrackingRegionEDProducerT(const edm::ParameterSet& iConfig)
      : regionsPutToken_{produces()}, regionProducer_(iConfig, consumesCollector()) {}

  ~TrackingRegionEDProducerT() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    T_TrackingRegionProducer::fillDescriptions(descriptions);
  }

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override {
    auto regions = regionProducer_.regions(iEvent, iSetup);
    iEvent.emplace(regionsPutToken_, std::move(regions));
  }

private:
  edm::EDPutTokenT<std::vector<std::unique_ptr<TrackingRegion>>> regionsPutToken_;
  T_TrackingRegionProducer regionProducer_;
};

#endif
