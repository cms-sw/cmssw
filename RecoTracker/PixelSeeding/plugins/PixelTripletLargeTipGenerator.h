#ifndef PixelTripletLargeTipGenerator_H
#define PixelTripletLargeTipGenerator_H

/** A HitTripletGenerator from HitPairGenerator and vector of
    Layers. The HitPairGenerator provides a set of hit pairs.
    For each pair the search for compatible hit(s) is done among
    provided Layers
 */

#include "CombinedHitTripletGenerator.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/PixelSeeding/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "RecoTracker/Record/interface/TrackerMultipleScatteringRecord.h"
#include "RecoTracker/TkMSParametrization/interface/MultipleScatteringParametrisationMaker.h"

#include <utility>
#include <vector>

class PixelTripletLargeTipGenerator : public HitTripletGeneratorFromPairAndLayers {
  typedef CombinedHitTripletGenerator::LayerCacheType LayerCacheType;

public:
  PixelTripletLargeTipGenerator(const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC)
      : PixelTripletLargeTipGenerator(cfg, iC) {}
  PixelTripletLargeTipGenerator(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);

  ~PixelTripletLargeTipGenerator() override;

  static void fillDescriptions(edm::ParameterSetDescription& desc);
  static const char* fillDescriptionsLabel() { return "pixelTripletLargeTip"; }

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
  const bool useFixedPreFiltering;
  const float extraHitRZtolerance;
  const float extraHitRPhitolerance;
  const bool useMScat;
  const bool useBend;
  const float dphi;

  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyESToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> fieldESToken_;
  edm::ESGetToken<MultipleScatteringParametrisationMaker, TrackerMultipleScatteringRecord> msmakerESToken_;
};
#endif
