#include "TrackingTools/PatternTools/interface/TempTrajectory.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "FWCore/Utilities/interface/Exception.h"


TempTrajectory::TempTrajectory(Trajectory && traj):
  theChiSquared(0),
  theNumberOfFoundHits(0), theNumberOfLostHits(0),
  theDirection(traj.direction()), theDirectionValidity(true),
  theValid(traj.isValid()),
  theNLoops(traj.nLoops()),
  theDPhiCache(traj.dPhiCacheForLoopersReconstruction()) {

  Trajectory::DataContainer::const_iterator begin=traj.measurements().begin();
  Trajectory::DataContainer::const_iterator end=traj.measurements().end();

  for(Trajectory::DataContainer::const_iterator it=begin; it!=end; ++it){
    push(std::move(*it));
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
    if (theData.front().updatedState().globalPosition().perp2() <
	theData.back().updatedState().globalPosition().perp2())
      theDirection = alongMomentum;
    else theDirection = oppositeToMomentum;
    theDirectionValidity = true;
  }
}

void TempTrajectory::push(const TempTrajectory& segment) {
  assert (segment.theDirection == theDirection) ;
  assert(theDirectionValidity); // given the above...

  const int N = segment.measurements().size();
  TrajectoryMeasurement const * tmp[N];
  int i=0;
  //for (DataContainer::const_iterator it = segment.measurements().rbegin(), ed = segment.measurements().rend(); it != ed; --it)
  for ( auto const & tm : segment.measurements())
    tmp[i++] =&tm;
  while(i!=0) theData.push_back(*tmp[--i]);
  theNumberOfFoundHits+= segment.theNumberOfFoundHits;
  theNumberOfLostHits += segment.theNumberOfLostHits;
  theChiSquared += segment.theChiSquared;
}

void TempTrajectory::join( TempTrajectory& segment) {
  assert (segment.theDirection == theDirection) ;
  assert(theDirectionValidity);

  if (segment.theData.shared()) {
    push(segment); 
    segment.theData.clear(); // obey the contract, and increase the chances it will be not shared one day
  } else {
    theData.join(segment.theData);
    theNumberOfFoundHits+= segment.theNumberOfFoundHits;
    theNumberOfLostHits += segment.theNumberOfLostHits;
    theChiSquared += segment.theChiSquared;
  }
}


PropagationDirection TempTrajectory::direction() const {
  if (theDirectionValidity) return PropagationDirection(theDirection);
  else throw cms::Exception("TrackingTools/PatternTools","Trajectory::direction() requested but not set");
}

void TempTrajectory::check() const {
  if ( theData.size() == 0) 
    throw cms::Exception("TrackingTools/PatternTools","Trajectory::check() - information requested from empty Trajectory");
}

bool TempTrajectory::lost( const TrackingRecHit& hit)
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
  assert(theDirectionValidity);
  PropagationDirection p=PropagationDirection(theDirection);
  Trajectory traj(p);
  traj.setNLoops(theNLoops);

  traj.reserve(theData.size());
  const TrajectoryMeasurement* tmp[theData.size()];
  int i=0;
  for (DataContainer::const_iterator it = theData.rbegin(), ed = theData.rend(); it != ed; --it)
    tmp[i++] = &(*it);
  while(i!=0) traj.push(*tmp[--i]);
  return traj;
}

