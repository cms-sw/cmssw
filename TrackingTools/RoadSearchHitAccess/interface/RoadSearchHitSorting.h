#ifndef TrackingTools_RoadSearchHitSorting_H
#define TrackingTools_RoadSearchHitSorting_H

//
// Package:         TrackingTools/RoadSearchHitSorting
// Class:           SortHitsByGlobalPosition
//                  SortHitPointersByGLobalPosition
// 
// Description:     various sortings for TrackingRecHits
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Fri Jul  6 13:37:38 UTC 2007
//
// $Author: gutsche $
// $Date: 2007/06/29 23:54:04 $
// $Revision: 1.39 $
//

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

class SortHitsByGlobalPosition {
public:

  SortHitsByGlobalPosition( const TrackingGeometry * geometry_, 
			    PropagationDirection dir = alongMomentum) :
    geometry(geometry_), theDir(dir){}
  
  
  bool operator()( const TrackingRecHit& a, const TrackingRecHit& b) const {
    if (theDir == alongMomentum) return insideOutLess( a, b);
    else return insideOutLess( b, a);
  }
  
 private:

  bool insideOutLess(  const TrackingRecHit& a, const TrackingRecHit& b) const;
  
  bool barrelForwardLess(  const TrackingRecHit& a, const TrackingRecHit& b) const;
  
  const TrackingGeometry * geometry;
  PropagationDirection theDir;
};

class SortHitPointersByGlobalPosition {
public:

  SortHitPointersByGlobalPosition( const TrackingGeometry * geometry_, 
			    PropagationDirection dir = alongMomentum) :
    geometry(geometry_), theDir(dir){}
  
  
  bool operator()( const TrackingRecHit* a, const TrackingRecHit* b) const {
    if (theDir == alongMomentum) return insideOutLess( a, b);
    else return insideOutLess( b, a);
  }
  
 private:

  bool insideOutLess(  const TrackingRecHit* a, const TrackingRecHit* b) const;
  
  bool barrelForwardLess(  const TrackingRecHit* a, const TrackingRecHit* b) const;
  
  const TrackingGeometry * geometry;
  PropagationDirection theDir;
};

class SortHitTrajectoryPairsByGlobalPosition {
 public:
  
  SortHitTrajectoryPairsByGlobalPosition(){ };

  bool operator()(const std::pair<TransientTrackingRecHit::ConstRecHitPointer, TrajectoryMeasurement*> &HitTM1 ,
		  const std::pair<TransientTrackingRecHit::ConstRecHitPointer, TrajectoryMeasurement*> &HitTM2 ) const
  {
    return
      InsideOutCompare(HitTM1,HitTM2);
  }  
 

 private:

   bool InsideOutCompare( const std::pair<TransientTrackingRecHit::ConstRecHitPointer, TrajectoryMeasurement*> &HitTM1 ,
			  const std::pair<TransientTrackingRecHit::ConstRecHitPointer, TrajectoryMeasurement*> &HitTM2 )  const;

};

class SortHitsByY {
 public:
  SortHitsByY(const TrackerGeometry& tracker) : _tracker(tracker) {}
  bool operator()( const TrackingRecHit& rh1,
		   const TrackingRecHit& rh2) const
  {
    bool result = 
      static_cast<unsigned int>(std::abs(_tracker.idToDet(rh1.geographicalId())->surface().toGlobal(rh1.localPosition()).y()) * 1E7) <
      static_cast<unsigned int>(std::abs(_tracker.idToDet(rh2.geographicalId())->surface().toGlobal(rh2.localPosition()).y()) * 1E7) ;
    return result;
  };

 private:
  const TrackerGeometry& _tracker;
};

class SortHitPointersByY {
 public:
  SortHitPointersByY(const TrackerGeometry& tracker):_tracker(tracker){}
  bool operator()( const TrackingRecHit* rh1,
		   const TrackingRecHit* rh2) const
  {
    bool result = 
      static_cast<unsigned int>(std::abs(_tracker.idToDet(rh1->geographicalId())->surface().toGlobal(rh1->localPosition()).y()) * 1E7) <
      static_cast<unsigned int>(std::abs(_tracker.idToDet(rh2->geographicalId())->surface().toGlobal(rh2->localPosition()).y()) * 1E7) ;
    return result;
  };

 private:
  const TrackerGeometry& _tracker;
};


#endif
