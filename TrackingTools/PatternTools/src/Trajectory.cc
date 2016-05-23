#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterTools.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "boost/intrusive_ptr.hpp" 
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h" 
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include <algorithm>

using namespace std;

void Trajectory::pop() {
  if (!empty()) {
    if(theData.back().recHit()->isValid()) {
      theNumberOfFoundHits--;
      theChiSquared -= theData.back().estimate();
    }
    else if(lost(* (theData.back().recHit()) )) {
      theNumberOfLostHits--;
    }
    else if(isBad(* (theData.back().recHit()) ) && theData.back().recHit()->geographicalId().det()==DetId::Muon ) {
      theChiSquaredBad -= theData.back().estimate();
    }
    else if(badForCCC(theData.back())) theNumberOfCCCBadHits_--;

    theData.pop_back();
  }
}


void Trajectory::push( const TrajectoryMeasurement& tm) {
  push( tm, tm.estimate());
}


void Trajectory::push(TrajectoryMeasurement && tm) {
  push( tm, tm.estimate());
}



void Trajectory::push(const TrajectoryMeasurement & tm, double chi2Increment) {
  theData.push_back(tm); pushAux(chi2Increment);
}

void Trajectory::push(TrajectoryMeasurement && tm, double chi2Increment) {
  theData.push_back(tm);  pushAux(chi2Increment);
}


void Trajectory::pushAux(double chi2Increment) {
 const TrajectoryMeasurement& tm = theData.back();
  if ( tm.recHit()->isValid()) {
    theChiSquared += chi2Increment;
    theNumberOfFoundHits++;
  }
  // else if (lost( tm.recHit()) && !inactive(tm.recHit().det())) theNumberOfLostHits++;
  else if (lost( *(tm.recHit()) ) ) {
    theNumberOfLostHits++;
  }
 
  else if (isBad( *(tm.recHit()) ) && tm.recHit()->geographicalId().det()==DetId::Muon ) {
    theChiSquaredBad += chi2Increment;
  }
 
  else if (badForCCC(tm)) theNumberOfCCCBadHits_++;

  // in case of a Trajectory constructed without direction, 
  // determine direction from the radii of the first two measurements

  if ( !theDirectionValidity && theData.size() >= 2) {
    if (theData[0].updatedState().globalPosition().perp2() <
	theData.back().updatedState().globalPosition().perp2())
      theDirection = alongMomentum;
    else theDirection = oppositeToMomentum;
    theDirectionValidity = true;
  }
}


int Trajectory::ndof(bool bon) const {
  Trajectory::RecHitContainer && transRecHits = recHits();
  
  int dof = 0;
  int dofBad = 0;
  
  for(Trajectory::RecHitContainer::const_iterator rechit = transRecHits.begin();
      rechit != transRecHits.end(); ++rechit) {
    if((*rechit)->isValid())
      dof += (*rechit)->dimension();
    else if( isBad(**rechit) && (*rechit)->geographicalId().det()==DetId::Muon )
      dofBad += (*rechit)->dimension();
  }

  // If dof!=0 (there is at least 1 valid hit),
  //    return ndof=ndof(fit)
  // If dof=0 (all rec hits are invalid, only for STA trajectories),
  //    return ndof=ndof(invalid hits)
  if(dof) {
    int constr = bon ? 5 : 4;
    return std::max(dof - constr, 0);
  }
  else {
    // A STA can have < 5 (invalid) hits
    // if this is the case ==> ndof = 1
    // (to avoid divisions by 0)
    int constr = bon ? 5 : 4;
    return std::max(dofBad - constr, 1);
  }
}


void Trajectory::validRecHits(ConstRecHitContainer & hits) const {
  hits.reserve(foundHits());
  for (Trajectory::DataContainer::const_iterator itm
	 = theData.begin(); itm != theData.end(); itm++)
    if ((*itm).recHit()->isValid()) hits.push_back((*itm).recHit());
}


PropagationDirection const & Trajectory::direction() const {
  if (theDirectionValidity) return theDirection;
  else throw cms::Exception("TrackingTools/PatternTools","Trajectory::direction() requested but not set");
}

void Trajectory::check() const {
  if ( theData.empty()) 
    throw cms::Exception("TrackingTools/PatternTools","Trajectory::check() - information requested from empty Trajectory");
}

bool Trajectory::lost( const TrackingRecHit& hit)
{
  if ( hit.isValid()) return false;
  else {
  //     // A DetLayer is always inactive in this logic.
  //     // The DetLayer is the Det of an invalid RecHit only if no DetUnit 
  //     // is compatible with the predicted state, so we don't really expect
  //     // a hit in this case.
  
    if(hit.geographicalId().rawId() == 0) {return false;}
    else{
      return hit.getType() == TrackingRecHit::missing;
    }
  }
}

bool Trajectory::isBad( const TrackingRecHit& hit)
{
  if ( hit.isValid()) return false;
  else {
    if(hit.geographicalId().rawId() == 0) {return false;}
    else{
      return hit.getType() == TrackingRecHit::bad;
    }
  }
}

bool Trajectory::badForCCC(const TrajectoryMeasurement &tm) {
  auto const * thit = dynamic_cast<const BaseTrackerRecHit*>( tm.recHit()->hit() );
  if (!thit)
    return false;
  if (thit->isPixel())
    return false;
  if (!tm.updatedState().isValid())
    return false;
  return siStripClusterTools::chargePerCM(thit->rawId(),
                                          thit->firstClusterRef().stripCluster(),
                                          tm.updatedState().localParameters()) < theCCCThreshold_;
}

void Trajectory::updateBadForCCC(float ccc_threshold) {
  // If the supplied threshold is the same as the currently cached
  // one, then return the current number of bad hits for CCC,
  // otherwise do a new full rescan.
  if (ccc_threshold == theCCCThreshold_)
    return;

  theCCCThreshold_ = ccc_threshold;
  theNumberOfCCCBadHits_ = 0;
  for (auto const & h : theData) {
    if (badForCCC(h))
      theNumberOfCCCBadHits_++;
  }
}

int Trajectory::numberOfCCCBadHits(float ccc_threshold) {
  updateBadForCCC(ccc_threshold);
  return theNumberOfCCCBadHits_;
}

TrajectoryStateOnSurface Trajectory::geometricalInnermostState() const {

  check();

  //if trajectory is in one half, return the end closer to origin point
  if ( firstMeasurement().updatedState().globalMomentum().perp() > 1.0
      && ( firstMeasurement().updatedState().globalPosition().basicVector().dot( firstMeasurement().updatedState().globalMomentum().basicVector() ) *
       lastMeasurement().updatedState().globalPosition().basicVector().dot( lastMeasurement().updatedState().globalMomentum().basicVector() )  > 0 ) ) {
     return (firstMeasurement().updatedState().globalPosition().mag() < lastMeasurement().updatedState().globalPosition().mag() ) ?
            firstMeasurement().updatedState() : lastMeasurement().updatedState();
  }

  //more complicated in case of traversing and low-pt trajectories with loops
  return closestMeasurement(GlobalPoint(0.0,0.0,0.0)).updatedState();

}


namespace {
  /// used to determine closest measurement to given point
  struct LessMag {
    LessMag(GlobalPoint point) : thePoint(point) {}
    bool operator()(const TrajectoryMeasurement& lhs,
                    const TrajectoryMeasurement& rhs) const{ 
      if (lhs.updatedState().isValid() && rhs.updatedState().isValid())
	return (lhs.updatedState().globalPosition() - thePoint).mag2() < (rhs.updatedState().globalPosition() -thePoint).mag2();
      else
	{
	  edm::LogError("InvalidStateOnMeasurement")<<"an updated state is not valid. result of LessMag comparator will be wrong.";
	  return false;
	}
    }
    GlobalPoint thePoint;
  };

}

TrajectoryMeasurement const & Trajectory::closestMeasurement(GlobalPoint point) const {
  check();
  vector<TrajectoryMeasurement>::const_iterator iter = std::min_element(measurements().begin(), measurements().end(), LessMag(point) );

  return (*iter);
}

void Trajectory::reverse() {
    // reverse the direction (without changing it if it's not along or opposite)
    if (theDirection == alongMomentum)           theDirection = oppositeToMomentum;
    else if (theDirection == oppositeToMomentum) theDirection = alongMomentum;
    // reverse the order of the hits
    std::reverse(theData.begin(), theData.end());
}
