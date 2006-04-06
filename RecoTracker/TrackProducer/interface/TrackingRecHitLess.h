#ifndef TrackCandidate_TrackingRecHitLess_H
#define TrackCandidate_TrackingRecHitLess_H

#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include <functional>
#include "Geometry/TrackerGeometryBuilder/interface/GeomDetLess.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

/** Defines order of layers in the Tracker as seen by straight tracks
 *  coming from the interaction region.
 */


class TrackingRecHitLess {
public:

  TrackingRecHitLess( const TrackingGeometry * geometry_, PropagationDirection dir = alongMomentum) :
    g_(geometry_){
    less_ = new GeomDetLess(dir);
  }


  bool operator()( const TrackingRecHit& a, const TrackingRecHit& b) const {
    
    return  less_->operator()(
			     g_->idToDet(a.geographicalId()), g_->idToDet(b.geographicalId()));
  }

 private:
  const TrackingGeometry * g_;
  GeomDetLess * less_;
};

#endif
