#ifndef RecoTracker_TkTrackingRegions_TrackingRegionEDProducerT_H
#define RecoTracker_TkTrackingRegions_TrackingRegionEDProducerT_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "DataFormats/Common/interface/OwnVector.h"

template <typename T_TrackingRegionProducer>
class TrackingRegionEDProducerT: public edm::stream::EDProducer<> {
public:
  // using OwnVector as vector<shared_ptr> and vector<unique_ptr> cause problems
  // I can't get dictionary compiled with unique_ptr
  // shared_ptr fails with runtime error "Class name 'TrackingRegionstdshared_ptrs' contains an underscore ('_'), which is illegal in the name of a product."
  using ProductType = edm::OwnVector<TrackingRegion>;

  TrackingRegionEDProducerT(const edm::ParameterSet& iConfig):
    regionProducer_(iConfig, consumesCollector()) {
    produces<ProductType>();
  }

  ~TrackingRegionEDProducerT() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    T_TrackingRegionProducer::fillDescriptions(descriptions);
  }

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override {
    auto regions = regionProducer_.regions(iEvent, iSetup);
    auto ret = std::make_unique<ProductType>();
    ret->reserve(regions.size());
    for(auto& regionPtr: regions) {
      ret->push_back(regionPtr.release());
    }

    iEvent.put(std::move(ret));
  }

private:
  T_TrackingRegionProducer regionProducer_;
};

#endif
