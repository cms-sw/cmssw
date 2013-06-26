#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "boost/intrusive_ptr.hpp" 
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h" 
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include <Geometry/CommonDetUnit/interface/GeomDetUnit.h>
#include <Geometry/CommonDetUnit/interface/GeomDetType.h>


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

    theData.pop_back();
  }
}


void Trajectory::push( const TrajectoryMeasurement& tm) {
  push( tm, tm.estimate());
}

void Trajectory::push( const TrajectoryMeasurement& tm, double chi2Increment)
{
  theData.push_back(tm);
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
 
  // in case of a Trajectory constructed without direction, 
  // determine direction from the radii of the first two measurements

  if ( !theDirectionValidity && theData.size() >= 2) {
    if (theData[0].updatedState().globalPosition().perp() <
	theData.back().updatedState().globalPosition().perp())
      theDirection = alongMomentum;
    else theDirection = oppositeToMomentum;
    theDirectionValidity = true;
  }
}

Trajectory::RecHitContainer Trajectory::recHits(bool splitting) const {
  RecHitContainer hits;
  recHitsV(hits,splitting);
  return hits;
}


int Trajectory::ndof(bool bon) const {
  const Trajectory::RecHitContainer transRecHits = recHits();
  
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



void Trajectory::recHitsV(ConstRecHitContainer & hits,bool splitting) const {
  hits.reserve(theData.size());
  if(!splitting){  
    for (Trajectory::DataContainer::const_iterator itm
	   = theData.begin(); itm != theData.end(); itm++){    
      hits.push_back((*itm).recHit());
    }
  }else{    
    for (Trajectory::DataContainer::const_iterator itm
	   = theData.begin(); itm != theData.end(); itm++){    

      // ====== WARNING: this is a temporary solution =========
      //        all this part of code should be implemented internally 
      //        in the TrackingRecHit classes. The concrete types of rechit 
      //        should be transparent to the Trajectory class

      if( typeid(*(itm->recHit()->hit())) == typeid(SiStripMatchedRecHit2D)){
      	LocalPoint firstLocalPos = 
	  itm->updatedState().surface().toLocal(itm->recHit()->transientHits()[0]->globalPosition());
	
	LocalPoint secondLocalPos = 
	  itm->updatedState().surface().toLocal(itm->recHit()->transientHits()[1]->globalPosition());
	
	LocalVector Delta = secondLocalPos - firstLocalPos;
	float scalar  = Delta.z() * (itm->updatedState().localDirection().z());
	

	TransientTrackingRecHit::ConstRecHitPointer hitA, hitB;

	// Get 2D strip Hits from a matched Hit.
 	//hitA = itm->recHit()->transientHits()[0];
 	//hitB = itm->recHit()->transientHits()[1];

	// Get 2D strip Hits from a matched Hit. Then get the 1D hit from the 2D hit
	if(!itm->recHit()->transientHits()[0]->detUnit()->type().isEndcap()){
	  hitA = itm->recHit()->transientHits()[0]->transientHits()[0];
	  hitB = itm->recHit()->transientHits()[1]->transientHits()[0];
	}else{ //don't use 1D hit in the endcap yet
	  hitA = itm->recHit()->transientHits()[0];
	  hitB = itm->recHit()->transientHits()[1];
	}

	if( (scalar>=0 && direction()==alongMomentum) ||
	    (scalar<0 && direction()==oppositeToMomentum)){
	  hits.push_back(hitA);
	  hits.push_back(hitB);
	}else if( (scalar>=0 && direction()== oppositeToMomentum) ||
		  (scalar<0 && direction()== alongMomentum)){
	  hits.push_back(hitB);
	  hits.push_back(hitA);
	}else {
	  //throw cms::Exception("Error in Trajectory::recHitsV(). Direction is not defined");	
          edm::LogError("Trajectory_recHitsV_UndefinedTrackDirection") 
            << "Error in Trajectory::recHitsV: scalar = " << scalar 
	    << ", direction = " << (direction()==alongMomentum ? "along " : (direction()==oppositeToMomentum ? "opposite " : "undefined ")) 
	    << theDirection <<"\n";
          hits.push_back(hitA);
          hits.push_back(hitB);
        }         
      }else if(typeid(*(itm->recHit()->hit())) == typeid(ProjectedSiStripRecHit2D)){
	//hits.push_back(itm->recHit()->transientHits()[0]);	//Use 2D SiStripRecHit
	if(!itm->recHit()->transientHits()[0]->detUnit()->type().isEndcap()){
	  hits.push_back(itm->recHit()->transientHits()[0]->transientHits()[0]);	//Use 1D SiStripRecHit
	}else{
	  hits.push_back(itm->recHit()->transientHits()[0]);	//Use 2D SiStripRecHit
	}
	// ===================================================================================	
      }else if(typeid(*(itm->recHit()->hit())) == typeid(SiStripRecHit2D)){
	//hits.push_back(itm->recHit());  //Use 2D SiStripRecHit
	if(!itm->recHit()->detUnit()->type().isEndcap()){
	  hits.push_back(itm->recHit()->transientHits()[0]); //Use 1D SiStripRecHit
	}else{
	  hits.push_back(itm->recHit());  //Use 2D SiStripRecHit
	}
      }else{
	hits.push_back(itm->recHit());
      }
    }//end loop on measurements
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

bool Trajectory::lost( const TransientTrackingRecHit& hit)
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

bool Trajectory::isBad( const TransientTrackingRecHit& hit)
{
  if ( hit.isValid()) return false;
  else {
    if(hit.geographicalId().rawId() == 0) {return false;}
    else{
      return hit.getType() == TrackingRecHit::bad;
    }
  }
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
