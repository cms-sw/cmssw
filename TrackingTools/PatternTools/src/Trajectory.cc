#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "FWCore/Utilities/interface/Exception.h"

void Trajectory::pop() {
  if (!empty()) {
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

edm::OwnVector<TransientTrackingRecHit> Trajectory::recHits() const {
  edm::OwnVector<TransientTrackingRecHit> hits;
  hits.reserve(theData.size());

  for (Trajectory::DataContainer::const_iterator itm
	 = theData.begin(); itm != theData.end(); itm++) {
    hits.push_back((*itm).recHit()->clone());
  }
  return hits;
}


int Trajectory::foundHits() const {
  int f = 0;
  for (edm::OwnVector<TransientTrackingRecHit>::const_iterator i=recHits().begin();
       i!=recHits().end();i++){
    if (i->isValid()) f++;
  }
  return f;
}

int Trajectory::lostHits() const {
  int l = 0;
  for (edm::OwnVector<TransientTrackingRecHit>::const_iterator i=recHits().begin();
       i!=recHits().end();i++){
    if (!(i->isValid())) l++;
  }
  return l;
}

PropagationDirection Trajectory::direction() const {
  if (theDirectionValidity) return theDirection;
  else throw cms::Exception("TrackingTools/PatternTools","Trajectory::direction() requested but not set");
}

void Trajectory::check() const {
  if ( theData.empty()) 
    throw cms::Exception("TrackingTools/PatternTools","Trajectory::check() - information requested from empty Trajectory");
}

bool Trajectory::lost( const TransientTrackingRecHit& hit)
{
  // FIXME
  if ( hit.isValid()) return false;
  //  else {
  //     // A DetLayer is always inactive in this logic.
  //     // The DetLayer is the Det of an invalid RecHit only if no DetUnit 
  //     // is compatible with the predicted state, so we don't really expect
  //     // a hit in this case.
  //     if ( dynamic_cast<const DetLayer*>( &hit.det()) != 0) return false;
  //     else {
  //       if (inactive(hit.det())) return false;
  else return true;
  //     }
  //  }
}

