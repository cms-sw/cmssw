#ifndef RecoTracker_PixelLowPtUtilities_StripSubClusterShapeTrajectoryFilter_h
#define RecoTracker_PixelLowPtUtilities_StripSubClusterShapeTrajectoryFilter_h

#include <vector>
#include <unordered_map>
#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"

class ClusterShapeHitFilter;
class TrackerTopology;
class TrackerGeometry;
class TrajectoryMeasurement;
class TrajectoryStateOnSurface;
class MeasurementTrackerEvent;
class SiStripNoises;
class TTree;
namespace edm {
  class Event;
  class EventSetup;
  class ConsumesCollector;
}  // namespace edm

//#define StripSubClusterShapeFilterBase_COUNTERS

class StripSubClusterShapeFilterBase {
public:
  StripSubClusterShapeFilterBase(const edm::ParameterSet &iConfig, edm::ConsumesCollector &iC);
  virtual ~StripSubClusterShapeFilterBase();

  static void fillPSetDescription(edm::ParameterSetDescription &iDesc);

protected:
  void setEventBase(const edm::Event &, const edm::EventSetup &);

  bool testLastHit(const TrackingRecHit *hit, const TrajectoryStateOnSurface &tsos, bool mustProject = false) const;
  bool testLastHit(const TrackingRecHit *hit,
                   const GlobalPoint &gpos,
                   const GlobalVector &gdir,
                   bool mustProject = false) const;

  // esConsumes tokens
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  edm::ESGetToken<ClusterShapeHitFilter, CkfComponentsRecord> csfToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  edm::ESGetToken<SiStripNoises, SiStripNoisesRcd> stripNoiseToken_;

  // who am i
  std::string label_;

  // pass-through of clusters with too many consecutive saturated strips
  uint32_t maxNSat_;

  // trimming parameters
  uint8_t trimMaxADC_;
  float trimMaxFracTotal_, trimMaxFracNeigh_;

  // maximum difference after peak finding
  float maxTrimmedSizeDiffPos_, maxTrimmedSizeDiffNeg_;

  // peak finding parameters
  float subclusterWindow_;
  float seedCutMIPs_, seedCutSN_;
  float subclusterCutMIPs_, subclusterCutSN_;

  // layers in which to apply the filter
  std::array<std::array<uint8_t, 10>, 7> layerMask_;

#ifdef StripSubClusterShapeFilterBase_COUNTERS
  mutable uint64_t called_, saturated_, test_, passTrim_, failTooLarge_, passSC_, failTooNarrow_;
#endif

  edm::ESHandle<TrackerGeometry> theTracker;
  edm::ESHandle<ClusterShapeHitFilter> theFilter;
  edm::ESHandle<SiStripNoises> theNoise;
  edm::ESHandle<TrackerTopology> theTopology;
};

class StripSubClusterShapeTrajectoryFilter : public StripSubClusterShapeFilterBase, public TrajectoryFilter {
public:
  StripSubClusterShapeTrajectoryFilter(const edm::ParameterSet &iConfig, edm::ConsumesCollector &iC)
      : StripSubClusterShapeFilterBase(iConfig, iC) {}

  ~StripSubClusterShapeTrajectoryFilter() override {}

  static void fillPSetDescription(edm::ParameterSetDescription &iDesc) {
    StripSubClusterShapeFilterBase::fillPSetDescription(iDesc);
  }

  bool qualityFilter(const TempTrajectory &) const override;
  bool qualityFilter(const Trajectory &) const override;

  bool toBeContinued(TempTrajectory &) const override;
  bool toBeContinued(Trajectory &) const override;

  std::string name() const override { return "StripSubClusterShapeTrajectoryFilter"; }

  void setEvent(const edm::Event &e, const edm::EventSetup &es) override { setEventBase(e, es); }

protected:
  using StripSubClusterShapeFilterBase::testLastHit;
  bool testLastHit(const TrajectoryMeasurement &last) const;
};

class StripSubClusterShapeSeedFilter : public StripSubClusterShapeFilterBase, public SeedComparitor {
public:
  StripSubClusterShapeSeedFilter(const edm::ParameterSet &iConfig, edm::ConsumesCollector &iC);

  ~StripSubClusterShapeSeedFilter() override {}

  void init(const edm::Event &ev, const edm::EventSetup &es) override { setEventBase(ev, es); }
  // implemented
  bool compatible(const TrajectoryStateOnSurface &tsos, SeedingHitSet::ConstRecHitPointer hit) const override;
  // not implemented
  bool compatible(const SeedingHitSet &hits) const override { return true; }
  bool compatible(const SeedingHitSet &hits,
                  const GlobalTrajectoryParameters &helixStateAtVertex,
                  const FastHelix &helix) const override;

protected:
  bool filterAtHelixStage_;
};

#endif
