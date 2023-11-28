#ifndef RecoTracker_PixelSeeding_plugins_PixelTripletNoTipGenerator_h
#define RecoTracker_PixelSeeding_plugins_PixelTripletNoTipGenerator_h

#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/PixelSeeding/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "RecoTracker/Record/interface/TrackerMultipleScatteringRecord.h"
#include "RecoTracker/TkMSParametrization/interface/MultipleScatteringParametrisationMaker.h"

#include "CombinedHitTripletGenerator.h"

#include <utility>
#include <vector>

class PixelTripletNoTipGenerator : public HitTripletGeneratorFromPairAndLayers {
  typedef CombinedHitTripletGenerator::LayerCacheType LayerCacheType;

public:
  PixelTripletNoTipGenerator(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);

  ~PixelTripletNoTipGenerator() override;

  void hitTriplets(const TrackingRegion& region,
                   OrderedHitTriplets& trs,
                   const edm::Event& ev,
                   const edm::EventSetup& es,
                   const SeedingLayerSetsHits::SeedingLayerSet& pairLayers,
                   const std::vector<SeedingLayerSetsHits::SeedingLayer>& thirdLayers) override;
  void hitTriplets(const TrackingRegion& region,
                   OrderedHitTriplets& result,
                   const edm::EventSetup& es,
                   const HitDoublets& doublets,
                   const RecHitsSortedInPhi** thirdHitMap,
                   const std::vector<const DetLayer*>& thirdLayerDetLayer,
                   const int nThirdLayers) override;

private:
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> fieldToken_;
  edm::ESGetToken<MultipleScatteringParametrisationMaker, TrackerMultipleScatteringRecord> msmakerToken_;
  float extraHitRZtolerance;
  float extraHitRPhitolerance;
  float extraHitPhiToleranceForPreFiltering;
  double theNSigma;
  double theChi2Cut;
};
#endif
