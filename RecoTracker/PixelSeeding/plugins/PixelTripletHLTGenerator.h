#ifndef RecoTracker_PixelSeeding_plugins_PixelTripletHLTGenerator_h
#define RecoTracker_PixelSeeding_plugins_PixelTripletHLTGenerator_h

/** A HitTripletGenerator from HitPairGenerator and vector of
    Layers. The HitPairGenerator provides a set of hit pairs.
    For each pair the search for compatible hit(s) is done among
    provided Layers
 */

#include "CombinedHitTripletGenerator.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/PixelSeeding/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "RecoTracker/Record/interface/TrackerMultipleScatteringRecord.h"
#include "RecoTracker/TkMSParametrization/interface/MultipleScatteringParametrisationMaker.h"

#include <utility>
#include <vector>

class SeedComparitor;

class PixelTripletHLTGenerator : public HitTripletGeneratorFromPairAndLayers {
  typedef CombinedHitTripletGenerator::LayerCacheType LayerCacheType;

public:
  PixelTripletHLTGenerator(const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC)
      : PixelTripletHLTGenerator(cfg, iC) {}
  PixelTripletHLTGenerator(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);

  ~PixelTripletHLTGenerator() override;

  static void fillDescriptions(edm::ParameterSetDescription& desc);
  static const char* fillDescriptionsLabel() { return "pixelTripletHLT"; }

  void hitTriplets(const TrackingRegion& region,
                   OrderedHitTriplets& trs,
                   const edm::Event& ev,
                   const edm::EventSetup& es,
                   const SeedingLayerSetsHits::SeedingLayerSet& pairLayers,
                   const std::vector<SeedingLayerSetsHits::SeedingLayer>& thirdLayers) override;

  void hitTriplets(const TrackingRegion& region,
                   OrderedHitTriplets& trs,
                   const edm::Event& ev,
                   const edm::EventSetup& es,
                   const HitDoublets& doublets,
                   const std::vector<SeedingLayerSetsHits::SeedingLayer>& thirdLayers,
                   std::vector<int>* tripletLastLayerIndex,
                   LayerCacheType& layerCache);

  void hitTriplets(const TrackingRegion& region,
                   OrderedHitTriplets& result,
                   const edm::EventSetup& es,
                   const HitDoublets& doublets,
                   const RecHitsSortedInPhi** thirdHitMap,
                   const std::vector<const DetLayer*>& thirdLayerDetLayer,
                   const int nThirdLayers) override;

  void hitTriplets(const TrackingRegion& region,
                   OrderedHitTriplets& result,
                   const edm::EventSetup& es,
                   const HitDoublets& doublets,
                   const RecHitsSortedInPhi** thirdHitMap,
                   const std::vector<const DetLayer*>& thirdLayerDetLayer,
                   const int nThirdLayers,
                   std::vector<int>* tripletLastLayerIndex);

private:
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> fieldToken_;
  edm::ESGetToken<MultipleScatteringParametrisationMaker, TrackerMultipleScatteringRecord> msmakerToken_;
  const bool useFixedPreFiltering;
  const float extraHitRZtolerance;
  const float extraHitRPhitolerance;
  const bool useMScat;
  const bool useBend;
  const float dphi;
  std::unique_ptr<SeedComparitor> theComparitor;
};
#endif
