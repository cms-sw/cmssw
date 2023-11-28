#ifndef SpecialSeedGenerators_GenericTripletGenerator_h
#define SpecialSeedGenerators_GenericTripletGenerator_h
//FWK
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/PixelSeeding/interface/OrderedHitTriplets.h"
#include "RecoTracker/TkSeedingLayers/interface/OrderedSeedingHits.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"

class GenericTripletGenerator : public OrderedHitsGenerator {
public:
  GenericTripletGenerator(const edm::ParameterSet& conf, edm::ConsumesCollector& iC);
  ~GenericTripletGenerator() override{};
  const OrderedSeedingHits& run(const TrackingRegion& region, const edm::Event& ev, const edm::EventSetup& es) override;
  void clear() override { hitTriplets.clear(); }

private:
  std::pair<bool, float> qualityFilter(const OrderedHitTriplet& oht,
                                       const std::map<float, OrderedHitTriplet>& map,
                                       const SeedingLayerSetsHits::SeedingLayerSet& ls) const;
  edm::EDGetTokenT<SeedingLayerSetsHits> theSeedingLayerToken;
  OrderedHitTriplets hitTriplets;
};

#endif
