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
  areAllReachableLayersSet(false),
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

SimpleForwardNavigableLayer::
SimpleForwardNavigableLayer( ForwardDetLayer* detLayer,
			     const BDLC& outerBL, 
                             const BDLC& allOuterBL,
                             const BDLC& innerBL,
                             const BDLC& allInnerBL,
			     const FDLC& outerFL, 
                             const FDLC& allOuterFL,
                             const FDLC& innerFL,
                             const FDLC& allInnerFL,
			     const MagneticField* field,
			     float epsilon,
			     bool checkCrossingSide) :
  SimpleNavigableLayer(field,epsilon,checkCrossingSide),
  areAllReachableLayersSet(true),
  theDetLayer(detLayer), 
  theOuterBarrelLayers(outerBL),
  theAllOuterBarrelLayers(allOuterBL),
  theInnerBarrelLayers(innerBL),
  theAllInnerBarrelLayers(allInnerBL),
  theOuterForwardLayers(outerFL),
  theAllOuterForwardLayers(allOuterFL),
  theInnerForwardLayers(innerFL),
  theAllInnerForwardLayers(allInnerFL),
  theOuterLayers(0), 
  theInnerLayers(0),
  theAllOuterLayers(0),
  theAllInnerLayers(0)
{
  
  // put barrel and forward layers together
  theOuterLayers.reserve(outerBL.size() + outerFL.size());
  for ( ConstBDLI bl = outerBL.begin(); bl != outerBL.end(); bl++ ) 
    theOuterLayers.push_back(*bl);
  for ( ConstFDLI fl = outerFL.begin(); fl != outerFL.end(); fl++ ) 
    theOuterLayers.push_back(*fl);

  theAllOuterLayers.reserve(allOuterBL.size() + allOuterFL.size());
  for ( ConstBDLI bl = allOuterBL.begin(); bl != allOuterBL.end(); bl++ )
    theAllOuterLayers.push_back(*bl);
  for ( ConstFDLI fl = allOuterFL.begin(); fl != allOuterFL.end(); fl++ )
    theAllOuterLayers.push_back(*fl);

  theInnerLayers.reserve(innerBL.size() + innerFL.size());
  for ( ConstBDLI bl = innerBL.begin(); bl != innerBL.end(); bl++ )
    theInnerLayers.push_back(*bl);
  for ( ConstFDLI fl = innerFL.begin(); fl != innerFL.end(); fl++ )
    theInnerLayers.push_back(*fl);

  theAllInnerLayers.reserve(allInnerBL.size() + allInnerFL.size());
  for ( ConstBDLI bl = allInnerBL.begin(); bl != allInnerBL.end(); bl++ )
    theAllInnerLayers.push_back(*bl);
  for ( ConstFDLI fl = allInnerFL.begin(); fl != allInnerFL.end(); fl++ )
    theAllInnerLayers.push_back(*fl);


  // sort the outer layers 
  sort(theOuterLayers.begin(), theOuterLayers.end(), TkLayerLess());
  sort(theInnerLayers.begin(), theInnerLayers.end(),TkLayerLess(outsideIn));
  sort(theOuterBarrelLayers.begin(), theOuterBarrelLayers.end(), TkLayerLess());
  sort(theInnerBarrelLayers.begin(), theInnerBarrelLayers.end(),TkLayerLess(outsideIn));
  sort(theOuterForwardLayers.begin(), theOuterForwardLayers.end(), TkLayerLess());
  sort(theInnerForwardLayers.begin(), theInnerForwardLayers.end(),TkLayerLess(outsideIn));

  sort(theAllOuterLayers.begin(), theAllOuterLayers.end(), TkLayerLess());
  sort(theAllInnerLayers.begin(), theAllInnerLayers.end(),TkLayerLess(outsideIn));

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

  //establish whether the tracks is crossing the tracker from outer layers to inner ones 
  //or from inner to outer
  //bool isInOutTrack  = (fts.position().basicVector().dot(fts.momentum().basicVector())>0) ? 1 : 0;
  float zpos = fts.position().z();
  bool isInOutTrackFWD = fts.momentum().z()*zpos>0;
  GlobalVector transversePosition(fts.position().x(), fts.position().y(), 0);
  bool isInOutTrackBarrel  = (transversePosition.dot(fts.momentum())>0) ? 1 : 0;	

  //establish whether inner or outer layers are crossed after propagation, according
  //to BOTH propagationDirection AND track momentum
  bool dirOppositeXORisInOutTrackBarrel = ( !(dir == oppositeToMomentum) && isInOutTrackBarrel) || ( (dir == oppositeToMomentum) && !isInOutTrackBarrel);
  bool dirOppositeXORisInOutTrackFWD = ( !(dir == oppositeToMomentum) && isInOutTrackFWD) || ( (dir == oppositeToMomentum) && !isInOutTrackFWD);
  //bool dirOppositeXORisInOutTrack = ( !(dir == oppositeToMomentum) && isInOutTrack) || ( (dir == oppositeToMomentum) && !isInOutTrack);

  if ( dirOppositeXORisInOutTrackFWD && dirOppositeXORisInOutTrackBarrel ) { //standard tracks

    //wellInside(ftsWithoutErrors, dir, theOuterForwardLayers.begin(), theOuterForwardLayers.end(), result);
    wellInside(ftsWithoutErrors, dir, theOuterLayers, result);

  }
  else if (!dirOppositeXORisInOutTrackFWD && !dirOppositeXORisInOutTrackBarrel){ // !dirOppositeXORisInOutTrack

    //wellInside(ftsWithoutErrors, dir, theInnerForwardLayers.begin(), theInnerForwardLayers.end(), result);
    wellInside(ftsWithoutErrors, dir, theInnerLayers, result);

  } else if (!dirOppositeXORisInOutTrackFWD && dirOppositeXORisInOutTrackBarrel ) {
    wellInside(ftsWithoutErrors, dir, theInnerForwardLayers.begin(), theInnerForwardLayers.end(), result);
    wellInside(ftsWithoutErrors, dir, theOuterBarrelLayers.begin(), theOuterBarrelLayers.end(), result);		

  } else {
    wellInside(ftsWithoutErrors, dir, theInnerBarrelLayers.begin(), theInnerBarrelLayers.end(), result);	
    wellInside(ftsWithoutErrors, dir, theOuterForwardLayers.begin(), theOuterForwardLayers.end(), result);

  }
/*
  if ( dirOppositeXORisInOutTrackBarrel ) {

    wellInside(ftsWithoutErrors, dir, theOuterBarrelLayers.begin(), theOuterBarrelLayers.end(), result);

  }
  else { // !dirOppositeXORisInOutTrack

    wellInside(ftsWithoutErrors, dir, theInnerBarrelLayers.begin(),theInnerBarrelLayers.end(), result);

  }
*/

  return result;
}


vector<const DetLayer*> 
SimpleForwardNavigableLayer::compatibleLayers( NavigationDirection dir) const
{
  if( !areAllReachableLayersSet ){
    edm::LogError("TkNavigation") << "ERROR: compatibleLayers() method used without all reachableLayers are set" ;
    throw DetLayerException("compatibleLayers() method used without all reachableLayers are set"); 
  }

  vector<const DetLayer*> result;

  if ( dir == insideOut ) {
    return theAllOuterLayers;
  }
  else {
    return theAllInnerLayers;
  }
  return result;

}

vector<const DetLayer*> 
SimpleForwardNavigableLayer::compatibleLayers( const FreeTrajectoryState& fts, 
					       PropagationDirection dir) const
{
  if likely( !areAllReachableLayersSet ){
    int counter = 0;
    return SimpleNavigableLayer::compatibleLayers(fts,dir,counter);
    //    edm::LogError("TkNavigation") << "ERROR: compatibleLayers() method used without all reachableLayers are set" ;
    //    throw DetLayerException("compatibleLayers() method used without all reachableLayers are set"); 
  }

  vector<const DetLayer*> result;
  FreeTrajectoryState ftsWithoutErrors = (fts.hasError()) ?
    FreeTrajectoryState(fts.parameters()) : fts;

  //establish whether the tracks is crossing the tracker from outer layers to inner ones 
  //or from inner to outer.
  //bool isInOutTrack  = (fts.position().basicVector().dot(fts.momentum().basicVector())>0) ? 1 : 0;
/*  float zpos = fts.position().z();
  bool isInOutTrack = fts.momentum().z()*zpos>0;
  
  //establish whether inner or outer layers are crossed after propagation, according
  //to BOTH propagationDirection AND track momentum
  bool dirOppositeXORisInOutTrack = ( !(dir == oppositeToMomentum) && isInOutTrack) || ((dir == oppositeToMomentum) && !isInOutTrack);

  if ( dirOppositeXORisInOutTrack ) {
    wellInside(ftsWithoutErrors, dir, theAllOuterLayers, result);
  }
  else { // !dirOppositeXORisInOutTrack
    wellInside(ftsWithoutErrors, dir, theAllInnerLayers, result);
  }
*/

  float zpos = fts.position().z();
  bool isInOutTrackFWD = fts.momentum().z()*zpos>0;
  GlobalVector transversePosition(fts.position().x(), fts.position().y(), 0);
  bool isInOutTrackBarrel  = (transversePosition.dot(fts.momentum())>0) ? 1 : 0;

  //establish whether inner or outer layers are crossed after propagation, according
  //to BOTH propagationDirection AND track momentum
  bool dirOppositeXORisInOutTrackBarrel = ( !(dir == oppositeToMomentum) && isInOutTrackBarrel) || ( (dir == oppositeToMomentum) && !isInOutTrackBarrel);
  bool dirOppositeXORisInOutTrackFWD = ( !(dir == oppositeToMomentum) && isInOutTrackFWD) || ( (dir == oppositeToMomentum) && !isInOutTrackFWD);
  //bool dirOppositeXORisInOutTrack = ( !(dir == oppositeToMomentum) && isInOutTrack) || ( (dir == oppositeToMomentum) && !isInOutTrack);

  if ( dirOppositeXORisInOutTrackFWD && dirOppositeXORisInOutTrackBarrel ) { //standard tracks

    //wellInside(ftsWithoutErrors, dir, theOuterForwardLayers.begin(), theOuterForwardLayers.end(), result);
    wellInside(ftsWithoutErrors, dir, theAllOuterLayers, result);

  }
  else if (!dirOppositeXORisInOutTrackFWD && !dirOppositeXORisInOutTrackBarrel){ // !dirOppositeXORisInOutTrack
  
    //wellInside(ftsWithoutErrors, dir, theInnerForwardLayers.begin(), theInnerForwardLayers.end(), result);
    wellInside(ftsWithoutErrors, dir, theAllInnerLayers, result);
  
  } else if (!dirOppositeXORisInOutTrackFWD && dirOppositeXORisInOutTrackBarrel ) {
        
    wellInside(ftsWithoutErrors, dir, theAllInnerForwardLayers.begin(), theAllInnerForwardLayers.end(), result);
    wellInside(ftsWithoutErrors, dir, theAllOuterBarrelLayers.begin(), theAllOuterBarrelLayers.end(), result);
  
  } else { 
  
    wellInside(ftsWithoutErrors, dir, theAllInnerBarrelLayers.begin(), theAllInnerBarrelLayers.end(), result);
    wellInside(ftsWithoutErrors, dir, theAllOuterForwardLayers.begin(), theAllOuterForwardLayers.end(), result);

  }
  return result;
}


DetLayer* SimpleForwardNavigableLayer::detLayer() const { return theDetLayer; }

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
		theAllOuterBarrelLayers.push_back(badditional);
		theOuterLayers.push_back(badditional);
		theAllOuterLayers.push_back(badditional);
                return;
        }
        theInnerBarrelLayers.push_back(badditional);
	theAllInnerBarrelLayers.push_back(badditional);
	theInnerLayers.push_back(badditional);
	theAllInnerLayers.push_back(badditional);
        return;
  } else if (fadditional){
        if (direction==insideOut){
                theOuterForwardLayers.push_back(fadditional);
		theAllOuterForwardLayers.push_back(fadditional);	
		theOuterLayers.push_back(badditional);
		theAllOuterLayers.push_back(badditional);
                return;
        }
        theInnerForwardLayers.push_back(fadditional);
	theAllInnerForwardLayers.push_back(fadditional);
	theInnerLayers.push_back(badditional);
	theAllInnerLayers.push_back(badditional);
        return;
  }
  edm::LogError("TkNavigation") << "trying to add neither a ForwardDetLayer nor a BarrelDetLayer";
  return;
} 
