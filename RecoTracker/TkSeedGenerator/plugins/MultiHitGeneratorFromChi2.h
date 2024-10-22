#ifndef MultiHitGeneratorFromChi2_H
#define MultiHitGeneratorFromChi2_H

/** A MultiHitGenerator from HitPairGenerator and vector of
    Layers. The HitPairGenerator provides a set of hit pairs.
    For each pair the search for compatible hit(s) is done among
    provided Layers
 */
#include "FWCore/Utilities/interface/Visibility.h"

#include "RecoTracker/TkSeedGenerator/interface/MultiHitGenerator.h"
#include "CombinedMultiHitGenerator.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "RecoTracker/TkSeedGenerator/interface/MultiHitGeneratorFromPairAndLayers.h"
#include "RecoTracker/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackingRecHit/interface/mayown_ptr.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/UniformEngine/interface/UniformMagneticField.h"

#include <utility>
#include <vector>

class HitPairGeneratorFromLayerPair;

class dso_hidden MultiHitGeneratorFromChi2 final : public MultiHitGeneratorFromPairAndLayers {
  typedef CombinedMultiHitGenerator::LayerCacheType LayerCacheType;

public:
  MultiHitGeneratorFromChi2(const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC)
      : MultiHitGeneratorFromChi2(cfg, iC) {}
  MultiHitGeneratorFromChi2(const edm::ParameterSet& cfg, edm::ConsumesCollector&);

  ~MultiHitGeneratorFromChi2() override;

  static void fillDescriptions(edm::ParameterSetDescription& desc);
  static const char* fillDescriptionsLabel() { return "multiHitFromChi2"; }

  void initES(const edm::EventSetup& es) override;

  void hitSets(const TrackingRegion& region,
               OrderedMultiHits& trs,
               const edm::Event& ev,
               const edm::EventSetup& es,
               SeedingLayerSetsHits::SeedingLayerSet pairLayers,
               std::vector<SeedingLayerSetsHits::SeedingLayer> thirdLayers) override;

  void hitSets(const TrackingRegion& region,
               OrderedMultiHits& trs,
               const HitDoublets& doublets,
               const std::vector<SeedingLayerSetsHits::SeedingLayer>& thirdLayers,
               LayerCacheType& layerCache,
               cacheHits& refittedHitStorage);

  void hitTriplets(const TrackingRegion& region,
                   OrderedMultiHits& result,
                   const HitDoublets& doublets,
                   const RecHitsSortedInPhi** thirdHitMap,
                   const std::vector<const DetLayer*>& thirdLayerDetLayer,
                   const int nThirdLayers) override;

  void hitSets(const TrackingRegion& region,
               OrderedMultiHits& result,
               const HitDoublets& doublets,
               const RecHitsSortedInPhi** thirdHitMap,
               const std::vector<const DetLayer*>& thirdLayerDetLayer,
               const int nThirdLayers,
               cacheHits& refittedHitStorage);

private:
  using HitOwnPtr = mayown_ptr<BaseTrackerRecHit>;

  void refit2Hits(HitOwnPtr& hit0,
                  HitOwnPtr& hit1,
                  TrajectoryStateOnSurface& tsos0,
                  TrajectoryStateOnSurface& tsos1,
                  const TrackingRegion& region,
                  float nomField,
                  bool isDebug);
  /*
  void refit3Hits(HitOwnPtr & hit0,
		  HitOwnPtr & hit1,
		  HitOwnPtr & hit2,
		  TrajectoryStateOnSurface& tsos0,
		  TrajectoryStateOnSurface& tsos1,
		  TrajectoryStateOnSurface& tsos2,
		  float nomField, bool isDebug);
  */
private:
  const ClusterShapeHitFilter* filter;
  TkTransientTrackingRecHitBuilder const* builder;
  TkClonerImpl cloner;

  bool useFixedPreFiltering;
  float extraHitRZtolerance;
  float extraHitRPhitolerance;
  float extraZKDBox;
  float extraRKDBox;
  float extraPhiKDBox;
  float dphi;
  const MagneticField* bfield;
  UniformMagneticField ufield = 0.;
  float nomField;
  double nSigmaRZ, nSigmaPhi, fnSigmaRZ;
  bool chi2VsPtCut;
  double maxChi2;
  std::vector<double> pt_interv;
  std::vector<double> chi2_cuts;
  bool refitHits;
  std::string filterName_;
  std::string builderName_;

  bool useSimpleMF_;
  std::string mfName_;

  std::vector<int> detIdsToDebug;

  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldESToken_;
  edm::ESGetToken<ClusterShapeHitFilter, CkfComponentsRecord> clusterShapeHitFilterESToken_;
  edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> transientTrackingRecHitBuilderESToken_;
};
#endif
