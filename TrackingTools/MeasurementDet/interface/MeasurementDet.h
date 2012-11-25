#ifndef MeasurementDet_H
#define MeasurementDet_H



#include "TrackingTools/MeasurementDet/interface/TempMeasurements.h"


#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"

#include "FWCore/Utilities/interface/GCC11Compatibility.h"


class TrajectoryStateOnSurface;
class Propagator;
class MeasurementEstimator;

class MeasurementDet {
public:
  typedef tracking::TempMeasurements TempMeasurements;
  typedef TransientTrackingRecHit::ConstRecHitContainer        RecHitContainer;

  MeasurementDet( const GeomDet* gdet) : theGeomDet(gdet) {}

  virtual RecHitContainer recHits( const TrajectoryStateOnSurface&) const = 0;

  // use a MeasurementEstimator to filter the hits (same algo as below..)
  // default as above 
  virtual bool recHits( const TrajectoryStateOnSurface& stateOnThisDet, const MeasurementEstimator&,
			RecHitContainer & result, std::vector<float> &) const {
    result = recHits(stateOnThisDet);
    return !result.empty();
  }

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
		    const MeasurementEstimator& est) const {

    TempMeasurements tmps;
    measurements(stateOnThisDet, est, tmps);
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
			     TempMeasurements & result) const =0;


  const GeomDet& fastGeomDet() const { return *theGeomDet;}
  virtual const GeomDet& geomDet() const { return *theGeomDet;}

  const BoundSurface& surface() const {return  geomDet().surface();}

  const Surface::PositionType& position() const {return geomDet().position();}

  virtual bool isActive() const=0;
  virtual bool hasBadComponents(const TrajectoryStateOnSurface &tsos) const=0;

 private:

  const GeomDet* theGeomDet;

};


#endif
