#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "boost/intrusive_ptr.hpp" 
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h" 
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"


using namespace std;

void Trajectory::pop() {
  if (!empty()) {
    if (theData.back().recHit()->isValid())             theNumberOfFoundHits--;
    else if(lost(* (theData.back().recHit()) )) theNumberOfLostHits--;
    theData.pop_back();
  }
}


// bool Trajectory::inactive( const Det& det) 
// {
//   typedef Det::DetUnitContainer DUC;

//   // DetUnit case -- straightforward
//   const DetUnit* detu = dynamic_cast<const DetUnit*>(&det);
//   if (detu != 0) return detu->inactive();
//   else {
//     const DetLayer* detl = dynamic_cast<const DetLayer*>(&det);
//     if (detl != 0) return false; // DetLayer should have inactive() too,
// 				 // but for the moment we skip it (see below)
//     else { // composite case
//       DUC duc = det.detUnits();
//       for (DUC::const_iterator i=duc.begin(); i!=duc.end(); i++) {
// 	if ( !(**i).inactive()) return false;
//       }
//       return true;
//     }
//   }
//   // the loop over DetUnits works for all 
//   // Dets, but it would be too slow for a big DetLayer; it would
//   // require creatind and copying the vector of all DetUnit* each time
//   // an invalid RecHit is produced by the layer, so it is penalizing
//   // even for active layers.
//   // Therefore the layer is not handled yet, and should eventually have
//   // it's own inactive() method.
// }
  
void Trajectory::push( const TrajectoryMeasurement& tm) {
  push( tm, tm.estimate());
}

void Trajectory::push( const TrajectoryMeasurement& tm, double chi2Increment)
{
  theData.push_back(tm);
  if ( tm.recHit()->isValid()) {
    theNumberOfFoundHits++;
   }
  //else if (lost( tm.recHit()) && !inactive(tm.recHit().det())) theNumberOfLostHits++;
  else if (lost( *(tm.recHit()) ) )   theNumberOfLostHits++;
  
 
  theChiSquared += chi2Increment;

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
  
  for(Trajectory::RecHitContainer::const_iterator rechit = transRecHits.begin();
      rechit != transRecHits.end(); ++rechit)
    if ((*rechit)->isValid()) dof += (*rechit)->dimension();
  
  int constr = bon ? 5 : 4;
  return std::max(dof - constr, 0); 
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
	
	if( (scalar>0 && direction()==alongMomentum) ||
	    (scalar<0 && direction()==oppositeToMomentum)){
	  hits.push_back(itm->recHit()->transientHits()[0]);
	  hits.push_back(itm->recHit()->transientHits()[1]);
	}else if( (scalar>0 && direction()== oppositeToMomentum) ||
		  (scalar<0 && direction()== alongMomentum)){
	  hits.push_back(itm->recHit()->transientHits()[1]);
	  hits.push_back(itm->recHit()->transientHits()[0]);
	}else
	  throw cms::Exception("Error in Trajectory::recHitsV(). Direction is not defined");	
      }else if(typeid(*(itm->recHit()->hit())) == typeid(ProjectedSiStripRecHit2D)){
	hits.push_back(itm->recHit()->transientHits()[0]);	
	// ===================================================================================	
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

