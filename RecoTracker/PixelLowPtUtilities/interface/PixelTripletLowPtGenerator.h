#ifndef _PixelTripletLowPtGenerator_h_
#define _PixelTripletLowPtGenerator_h_

/** A HitTripletGenerator from HitPairGenerator and vector of
    Layers. The HitPairGenerator provides a set of hit pairs.
    For each pair the search for compatible hit(s) is done among
    provided Layers
 */

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "RecoTracker/PixelSeeding/interface/HitTripletGeneratorFromPairAndLayers.h"

#include "RecoTracker/PixelLowPtUtilities/interface/TripletFilter.h"

class IdealMagneticFieldRecord;
class MultipleScatteringParametrisationMaker;
class TrackerMultipleScatteringRecord;
class SiPixelClusterShapeCache;
class TrackerGeometry;
class TransientTrackingRecHitBuilder;
class TransientRecHitRecord;
class TripletFilter;
class ClusterShapeHitFilter;
class CkfComponentsRecord;

#include <vector>

class PixelTripletLowPtGenerator : public HitTripletGeneratorFromPairAndLayers {
public:
  PixelTripletLowPtGenerator(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);

  ~PixelTripletLowPtGenerator() override;

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
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> m_geomToken;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> m_topoToken;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> m_magfieldToken;
  edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> m_ttrhBuilderToken;
  edm::ESGetToken<MultipleScatteringParametrisationMaker, TrackerMultipleScatteringRecord> m_msmakerToken;
  edm::ESGetToken<ClusterShapeHitFilter, CkfComponentsRecord> m_clusterFilterToken;

  void getTracker(const edm::EventSetup& es);
  GlobalPoint getGlobalPosition(const TrackingRecHit* recHit);

  const TrackerGeometry* theTracker;
  std::unique_ptr<TripletFilter> theFilter;

  edm::EDGetTokenT<SiPixelClusterShapeCache> theClusterShapeCacheToken;
  double nSigMultipleScattering;
  double rzTolerance;
  double maxAngleRatio;

  bool checkMultipleScattering;
  bool checkClusterShape;
};

#endif
