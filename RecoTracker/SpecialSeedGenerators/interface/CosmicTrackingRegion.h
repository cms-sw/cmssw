#ifndef CosmicTrackingRegion_H
#define CosmicTrackingRegion_H

/** \class CosmicTrackingRegion
 * A concrete implementation of TrackingRegion. 
 * Apart of vertex constraint from TrackingRegion in this implementation
 * the region of interest is further constrainted in phi and eta 
 * around the direction of the region 
 */

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionBase.h"
#include "RecoTracker/TkTrackingRegions/interface/HitRZConstraint.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackingRecHit/interface/mayown_ptr.h"

#include <vector>

using SeedingHit = BaseTrackerRecHit const*;

class CosmicTrackingRegion : public TrackingRegionBase {
public:
  ~CosmicTrackingRegion() override {}

  /** constructor (symmetric eta and phi margins). <BR>
   * dir        - the direction around which region is constructed <BR>
   *              the initial direction of the momentum of the particle 
   *              should be in the range <BR> 
   *              phi of dir +- deltaPhi <BR>
   *              eta of dir +- deltaEta <BR> 
   *              
   * vertexPos  - the position of the vertex (origin) of the of the region.<BR>
   *              It is a centre of cylinder constraind with rVertex, zVertex.
   *              The track of the particle should cross the cylinder <BR>
   *              WARNING: in the current implementaion the vertexPos is
   *              supposed to be placed on the beam line, i.e. to be of the form
   *              (0,0,float)
   *
   * ptMin      - minimal pt of interest <BR>
   * rVertex    - radius of the cylinder around beam line where the tracks
   *              of interest should point to. <BR>
   * zVertex    - half height of the cylinder around the beam line
   *              where the tracks of interest should point to.   <BR>
   * deltaEta   - allowed deviation of the initial direction of particle
   *              in eta in respect to direction of the region <BR>
   *  deltaPhi  - allowed deviation of the initial direction of particle
   *              in phi in respect to direction of the region 
  */
  CosmicTrackingRegion(const GlobalVector& dir,
                       const GlobalPoint& vertexPos,
                       float ptMin,
                       float rVertex,
                       float zVertex,
                       float deltaEta,
                       float deltaPhi,
                       float dummy = 0.,
                       const MeasurementTrackerEvent* measurementTracker = nullptr)
      : TrackingRegionBase(dir, vertexPos, Range(-1 / ptMin, 1 / ptMin), rVertex, zVertex),
        theMeasurementTracker_(measurementTracker),
        measurementTrackerName_("") {}

  CosmicTrackingRegion(const GlobalVector& dir,
                       const GlobalPoint& vertexPos,
                       float ptMin,
                       float rVertex,
                       float zVertex,
                       float deltaEta,
                       float deltaPhi,
                       const edm::ParameterSet& extra,
                       const MeasurementTrackerEvent* measurementTracker = nullptr)
      : TrackingRegionBase(dir, vertexPos, Range(-1 / ptMin, 1 / ptMin), rVertex, zVertex),
        theMeasurementTracker_(measurementTracker) {
    measurementTrackerName_ = extra.getParameter<std::string>("measurementTrackerName");
  }

  CosmicTrackingRegion(CosmicTrackingRegion const& rh)
      : TrackingRegionBase(rh),
        theMeasurementTracker_(rh.theMeasurementTracker_),
        measurementTrackerName_(rh.measurementTrackerName_) {}

  TrackingRegion::Hits hits(const edm::EventSetup& es, const SeedingLayerSetsHits::SeedingLayer& layer) const override;

  std::unique_ptr<HitRZCompatibility> checkRZ(const DetLayer* layer,
                                              const Hit& outerHit,
                                              const edm::EventSetup& iSetup,
                                              const DetLayer* outerlayer = nullptr,
                                              float lr = 0,
                                              float gz = 0,
                                              float dr = 0,
                                              float dz = 0) const override {
    return nullptr;
  }

  std::unique_ptr<TrackingRegion> clone() const override { return std::make_unique<CosmicTrackingRegion>(*this); }

  std::string name() const override { return "CosmicTrackingRegion"; }

private:
  template <typename T>
  void hits_(const edm::EventSetup& es, const T& layer, TrackingRegion::Hits& result) const;

  const MeasurementTrackerEvent* theMeasurementTracker_;
  std::string measurementTrackerName_;

  using cacheHitPointer = mayown_ptr<BaseTrackerRecHit>;
  using cacheHits = std::vector<cacheHitPointer>;

  // not a solution!  here just to try to get this thing working....
  // in any case onDemand is NOT thread safe yet
  // actually this solution is absolutely safe. It lays in the effimeral nature of the region itself
  mutable cacheHits cache;
};

#endif
