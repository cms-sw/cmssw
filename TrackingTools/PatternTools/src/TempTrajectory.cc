#include "TrackingTools/PatternTools/interface/TempTrajectory.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <ext/slist>

TempTrajectory::TempTrajectory( const Trajectory& traj):
  theSeed( traj.sharedSeed() ),
  theChiSquared(0),
  theNumberOfFoundHits(0), theNumberOfLostHits(0),
  theDirection(traj.direction()), theDirectionValidity(true),
  theValid(traj.isValid()),
  theNLoops(traj.nLoops()),
  theDPhiCache(traj.dPhiCacheForLoopersReconstruction()) {
  
  Trajectory::DataContainer::const_iterator begin=traj.measurements().begin();
  Trajectory::DataContainer::const_iterator end=traj.measurements().end();
  
  for(Trajectory::DataContainer::const_iterator it=begin; it!=end; ++it){
    push(*it);
  }

}


void TempTrajectory::pop() { 
  if (!empty()) {
    if (theData.back().recHit()->isValid())             theNumberOfFoundHits--;
    else if(lost(* (theData.back().recHit()) )) theNumberOfLostHits--;
    theData.pop_back();
  }
}



void TempTrajectory::pushAux(double chi2Increment) {
  const TrajectoryMeasurement& tm = theData.back();
  if ( tm.recHit()->isValid()) {
    theNumberOfFoundHits++;
   }
  //else if (lost( tm.recHit()) && !inactive(tm.recHit().det())) theNumberOfLostHits++;
  else if (lost( *(tm.recHit()) ) )   theNumberOfLostHits++;
  
  
  theChiSquared += chi2Increment;

  // in case of a Trajectory constructed without direction, 
  // determine direction from the radii of the first two measurements

  if ( !theDirectionValidity && theData.size() >= 2) {
    if (theData.front().updatedState().globalPosition().perp() <
	theData.back().updatedState().globalPosition().perp())
      theDirection = alongMomentum;
    else theDirection = oppositeToMomentum;
    theDirectionValidity = true;
  }
}

void TempTrajectory::push( const TempTrajectory& segment) {
  assert (segment.direction() == theDirection) ;
    __gnu_cxx::slist<const TrajectoryMeasurement*> list;
  for (DataContainer::const_iterator it = segment.measurements().rbegin(), ed = segment.measurements().rend(); it != ed; --it) {
        list.push_front(&(*it));
  }
  for(__gnu_cxx::slist<const TrajectoryMeasurement*>::const_iterator it = list.begin(), ed = list.end(); it != ed; ++it) {
        push(**it);
  }
}

void TempTrajectory::join( TempTrajectory& segment) {
  assert (segment.direction() == theDirection) ;
  if (segment.theData.shared()) {
      push(segment);
      segment.theData.clear(); // obey the contract, and increase the chances it will be not shared one day
  } else {
      for (DataContainer::const_iterator it = segment.measurements().rbegin(), ed = segment.measurements().rend(); it != ed; --it) {
          if ( it->recHit()->isValid())       theNumberOfFoundHits++;
          else if (lost( *(it->recHit()) ) ) theNumberOfLostHits++;
          theChiSquared += it->estimate();
      }
      theData.join(segment.theData);

      if ( !theDirectionValidity && theData.size() >= 2) {
        if (theData.front().updatedState().globalPosition().perp() <
            theData.back().updatedState().globalPosition().perp())
          theDirection = alongMomentum;
        else theDirection = oppositeToMomentum;
        theDirectionValidity = true;
      }
  }
}


/*
Trajectory::RecHitContainer Trajectory::recHits() const {
  RecHitContainer hits;
  hits.reserve(theData.size());

  for (Trajectory::DataContainer::const_iterator itm
	 = theData.begin(); itm != theData.end(); itm++) {
    hits.push_back((*itm).recHit());
  }
  return hits;
}

*/

PropagationDirection TempTrajectory::direction() const {
  if (theDirectionValidity) return PropagationDirection(theDirection);
  else throw cms::Exception("TrackingTools/PatternTools","Trajectory::direction() requested but not set");
}

void TempTrajectory::check() const {
  if ( theData.size() == 0) 
    throw cms::Exception("TrackingTools/PatternTools","Trajectory::check() - information requested from empty Trajectory");
}

bool TempTrajectory::lost( const TransientTrackingRecHit& hit)
{
  if  likely(hit.isValid()) return false;

  //     // A DetLayer is always inactive in this logic.
  //     // The DetLayer is the Det of an invalid RecHit only if no DetUnit 
  //     // is compatible with the predicted state, so we don't really expect
  //     // a hit in this case.
  
  if(hit.geographicalId().rawId() == 0) {return false;}
  return hit.getType() == TrackingRecHit::missing;
}

Trajectory TempTrajectory::toTrajectory() const {
  Trajectory traj(theSeed, PropagationDirection(theDirection));
  traj.setNLoops(theNLoops);

  traj.reserve(theData.size());
  static std::vector<const TrajectoryMeasurement*> work;
  work.resize(theData.size(), 0);
  std::vector<const TrajectoryMeasurement*>::iterator workend = work.end(), itwork = workend;
  for (TempTrajectory::DataContainer::const_iterator it = theData.rbegin(), ed = theData.rend(); it != ed; --it) {
       --itwork;  *itwork = (&(*it));
  }
  for (; itwork != workend; ++itwork) {
        traj.push(**itwork);
  }
  return traj;
}

