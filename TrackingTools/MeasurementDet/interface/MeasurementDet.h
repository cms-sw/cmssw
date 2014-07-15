#ifndef MeasurementDet_H
#define MeasurementDet_H

#include "TrackingTools/MeasurementDet/interface/TempMeasurements.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackingRecHit/interface/InvalidTrackingRecHit.h"

#include "FWCore/Utilities/interface/GCC11Compatibility.h"

class TrajectoryStateOnSurface;
class Propagator;
class MeasurementEstimator;
class MeasurementTrackerEvent;

class MeasurementDet {
public:
  typedef tracking::TempMeasurements TempMeasurements;
  typedef TrackingRecHit::ConstRecHitContainer        RecHitContainer;

  using SimpleHitContainer=std::vector<BaseTrackerRecHit *>;


  MeasurementDet( const GeomDet* gdet) : 
    theGeomDet(gdet), 
    theMissingHit(std::make_shared<InvalidTrackingRecHit>(fastGeomDet(),TrackingRecHit::missing)),
    theInactiveHit(std::make_shared<InvalidTrackingRecHit>(fastGeomDet(),TrackingRecHit::inactive)){}

  virtual RecHitContainer recHits( const TrajectoryStateOnSurface&, const MeasurementTrackerEvent &) const = 0;

  // use a MeasurementEstimator to filter the hits (same algo as below..)
  // default as above 
  virtual bool recHits( const TrajectoryStateOnSurface& stateOnThisDet, const MeasurementEstimator&, const MeasurementTrackerEvent & data,
			RecHitContainer & result, std::vector<float> &) const {
    result = recHits(stateOnThisDet, data);
    return !result.empty();
  }

  // default for non-tracker dets...
  virtual bool recHits(SimpleHitContainer & result,  
		       const TrajectoryStateOnSurface& stateOnThisDet, const MeasurementEstimator&, const MeasurementTrackerEvent & data) const { return false;}

  /** obsolete version in case the TrajectoryState on the surface of the
   *  Det is already available. The first TrajectoryStateOnSurface is on the surface of this 
   *  Det, and the second TrajectoryStateOnSurface is not used, as the propagator...
   * The stateOnThisDet should the result of <BR>
   *  prop.propagate( startingState, this->surface())
   */
  std::vector<TrajectoryMeasurement> 
  fastMeasurements( const TrajectoryStateOnSurface& stateOnThisDet, 
		    const TrajectoryStateOnSurface&, 
		    const Propagator&, 
		    const MeasurementEstimator& est,
                    const MeasurementTrackerEvent & data) const {

    TempMeasurements tmps;
    measurements(stateOnThisDet, est, data, tmps);
    std::vector<TrajectoryMeasurement> result;
    result.reserve(tmps.size());
    int index[tmps.size()];  tmps.sortIndex(index);
    for (std::size_t i=0; i!=tmps.size(); ++i) {
       auto j=index[i];
       result.emplace_back(stateOnThisDet,std::move(tmps.hits[j]),tmps.distances[j]);
    }
    return result;
  }

  // return false if missing ( if inactive is true and one hit)
  virtual bool measurements( const TrajectoryStateOnSurface& stateOnThisDet,
			     const MeasurementEstimator& est,
                             const MeasurementTrackerEvent & data,
			     TempMeasurements & result) const =0;


  const GeomDet& fastGeomDet() const { return *theGeomDet;}
  virtual const GeomDet& geomDet() const { return *theGeomDet;}

  const Surface& surface() const {return  geomDet().surface();}

  const Surface::PositionType& position() const {return geomDet().position();}

  virtual bool isActive(const MeasurementTrackerEvent & data) const=0;
  virtual bool hasBadComponents(const TrajectoryStateOnSurface &tsos, const MeasurementTrackerEvent & data) const=0;

 private:

  const GeomDet* theGeomDet;
protected:
  TrackingRecHit::ConstRecHitPointer theMissingHit;
  TrackingRecHit::ConstRecHitPointer theInactiveHit;

};


#endif
