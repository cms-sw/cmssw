#include "RecoTracker/TkNavigation/interface/SimpleForwardNavigableLayer.h"

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "TrackingTools/DetLayers/interface/DetLayerException.h"

#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"

#include "RecoTracker/TkNavigation/interface/TkLayerLess.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


using namespace std;

SimpleForwardNavigableLayer::
SimpleForwardNavigableLayer( ForwardDetLayer* detLayer,
			     const BDLC& outerBL, 
			     const FDLC& outerFL, 
			     const MagneticField* field,
			     float epsilon,
			     bool checkCrossingSide) :
  SimpleNavigableLayer(field,epsilon,checkCrossingSide),
  theDetLayer(detLayer), 
  theOuterBarrelLayers(outerBL),
  theInnerBarrelLayers(0),
  theOuterForwardLayers(outerFL),
  theInnerForwardLayers(0),
  theOuterLayers(0), 
  theInnerLayers(0)
{
  
  // put barrel and forward layers together
  theOuterLayers.reserve(outerBL.size() + outerFL.size());
  for ( ConstBDLI bl = outerBL.begin(); bl != outerBL.end(); bl++ ) 
    theOuterLayers.push_back(*bl);
  for ( ConstFDLI fl = outerFL.begin(); fl != outerFL.end(); fl++ ) 
    theOuterLayers.push_back(*fl);

  // sort the outer layers 
  sort(theOuterLayers.begin(), theOuterLayers.end(), TkLayerLess());
  sort(theOuterForwardLayers.begin(), theOuterForwardLayers.end(), TkLayerLess());
  sort(theOuterBarrelLayers.begin(), theOuterBarrelLayers.end(), TkLayerLess());

}




vector<const DetLayer*> 
SimpleForwardNavigableLayer::nextLayers( NavigationDirection dir) const
{
  vector<const DetLayer*> result;
  
  // the order is the one in which layers
  // should be checked for a reasonable trajectory

  if ( dir == insideOut ) {
    return theOuterLayers;
  }
  else {
    return theInnerLayers;
  }

  return result;
}

vector<const DetLayer*> 
SimpleForwardNavigableLayer::nextLayers( const FreeTrajectoryState& fts, 
					 PropagationDirection dir) const 
{
  // This method contains the sequence in which the layers are tested.
  // The iteration stops as soon as a layer contains the propagated state
  // within epsilon

  vector<const DetLayer*> result;

  FreeTrajectoryState ftsWithoutErrors = (fts.hasError()) ?
    FreeTrajectoryState(fts.parameters()) : fts;

  auto const position = fts.position();
  auto const momentum = fts.momentum();


  //establish whether the tracks is crossing the tracker from outer layers to inner ones 
  //or from inner to outer
  float zpos = position.z();
  bool isInOutTrackFWD =  momentum.z()*zpos>0;
  GlobalVector transversePosition(position.x(), position.y(), 0);
  bool isInOutTrackBarrel  = (transversePosition.dot(momentum)>0);	

  //establish whether inner or outer layers are crossed after propagation, according
  //to BOTH propagationDirection AND track momentum
  bool dirOppositeXORisInOutTrackBarrel = ( !(dir == oppositeToMomentum) && isInOutTrackBarrel) || ( (dir == oppositeToMomentum) && !isInOutTrackBarrel);
  bool dirOppositeXORisInOutTrackFWD = ( !(dir == oppositeToMomentum) && isInOutTrackFWD) || ( (dir == oppositeToMomentum) && !isInOutTrackFWD);
  //bool dirOppositeXORisInOutTrack = ( !(dir == oppositeToMomentum) && isInOutTrack) || ( (dir == oppositeToMomentum) && !isInOutTrack);

  if likely( dirOppositeXORisInOutTrackFWD && dirOppositeXORisInOutTrackBarrel ) { //standard tracks
    wellInside(ftsWithoutErrors, dir, theOuterLayers, result);
  }
  else if (!dirOppositeXORisInOutTrackFWD && !dirOppositeXORisInOutTrackBarrel){ // !dirOppositeXORisInOutTrack
    wellInside(ftsWithoutErrors, dir, theInnerLayers, result);
  } else if (!dirOppositeXORisInOutTrackFWD && dirOppositeXORisInOutTrackBarrel ) {
    wellInside(ftsWithoutErrors, dir, theInnerForwardLayers.begin(), theInnerForwardLayers.end(), result);
    wellInside(ftsWithoutErrors, dir, theOuterBarrelLayers.begin(), theOuterBarrelLayers.end(), result);		
  } else {
    wellInside(ftsWithoutErrors, dir, theInnerBarrelLayers.begin(), theInnerBarrelLayers.end(), result);	
    wellInside(ftsWithoutErrors, dir, theOuterForwardLayers.begin(), theOuterForwardLayers.end(), result);
  }

  return result;
}


vector<const DetLayer*> 
SimpleForwardNavigableLayer::compatibleLayers( NavigationDirection dir) const {
  edm::LogError("TkNavigation") << "ERROR: compatibleLayers() method used without all reachableLayers are set" ;
  throw DetLayerException("compatibleLayers() method used without all reachableLayers are set"); 
  return vector<const DetLayer*>();

}


void SimpleForwardNavigableLayer::setDetLayer( DetLayer* dl) {
  cerr << "Warning: SimpleForwardNavigableLayer::setDetLayer called."
       << endl << "This should never happen!" << endl;
}

void SimpleForwardNavigableLayer::setInwardLinks(const BDLC& innerBL, 
                                                 const FDLC& innerFL,
						 TkLayerLess sorter) {

  theInnerBarrelLayers  = innerBL;
  theInnerForwardLayers = innerFL;

  theInnerLayers.clear();
  theInnerLayers.reserve(innerBL.size() + innerFL.size());
  for ( ConstBDLI bl = innerBL.begin(); bl != innerBL.end(); bl++ )
    theInnerLayers.push_back(*bl);
  for ( ConstFDLI fl = innerFL.begin(); fl != innerFL.end(); fl++ )
    theInnerLayers.push_back(*fl);

  // sort the inner layers
  sort(theInnerLayers.begin(), theInnerLayers.end(),sorter);
  sort(theInnerForwardLayers.begin(), theInnerForwardLayers.end(),sorter);
  sort(theInnerBarrelLayers.begin(), theInnerBarrelLayers.end(), sorter);

}

void SimpleForwardNavigableLayer::setAdditionalLink(DetLayer* additional, NavigationDirection direction){
  ForwardDetLayer* fadditional = dynamic_cast<ForwardDetLayer*>(additional);
  BarrelDetLayer*  badditional = dynamic_cast<BarrelDetLayer*>(additional);
  if (badditional){
        if (direction==insideOut){
	  theOuterBarrelLayers.push_back(badditional);
	  theOuterLayers.push_back(badditional);
	  return;
        }
        theInnerBarrelLayers.push_back(badditional);
	theInnerLayers.push_back(badditional);
        return;
  } else if (fadditional){
    if (direction==insideOut){
      theOuterForwardLayers.push_back(fadditional);
      theOuterLayers.push_back(badditional);
      return;
    }
    theInnerForwardLayers.push_back(fadditional);
    theInnerLayers.push_back(badditional);
    return;
  }
  edm::LogError("TkNavigation") << "trying to add neither a ForwardDetLayer nor a BarrelDetLayer";
  return;
} 
