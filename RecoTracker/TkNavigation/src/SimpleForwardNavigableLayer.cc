#include "RecoTracker/TkNavigation/interface/SimpleForwardNavigableLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "Geometry/Surface/interface/BoundCylinder.h"
#include "Geometry/Surface/interface/BoundDisk.h"
#include "RecoTracker/TkNavigation/interface/TkLayerLess.h"
#include "TrackingTools/DetLayers/interface/DetLayerException.h"

using namespace std;

SimpleForwardNavigableLayer::
SimpleForwardNavigableLayer( ForwardDetLayer* detLayer,
			     const BDLC& outerBL, 
			     const FDLC& outerFL, 
			     const MagneticField* field,
			     float epsilon) :
  SimpleNavigableLayer(field,epsilon),
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
			     float epsilon) :
  SimpleNavigableLayer(field,epsilon),
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
  sort(theInnerLayers.begin(), theInnerLayers.end(),TkLayerLess(oppositeToMomentum));

  sort(theAllOuterLayers.begin(), theAllOuterLayers.end(), TkLayerLess());
  sort(theAllInnerLayers.begin(), theAllInnerLayers.end(),TkLayerLess(oppositeToMomentum));

}



vector<const DetLayer*> 
SimpleForwardNavigableLayer::nextLayers( PropagationDirection dir) const
{
  vector<const DetLayer*> result;
  
  // the order is the one in which layers
  // should be checked for a reasonable trajectory

  if ( dir == alongMomentum ) {
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

  if ( dir == alongMomentum ) {

    wellInside(ftsWithoutErrors, dir, theOuterLayers, result);

  }
  else { // oppositeToMomentum

    wellInside(ftsWithoutErrors, dir, theInnerLayers, result);

  }
  return result;
}


vector<const DetLayer*> 
SimpleForwardNavigableLayer::compatibleLayers( PropagationDirection dir) const
{
  if( !areAllReachableLayersSet ){
    cout << "ERROR: compatibleLayers() method used without all reachableLayers are set" << endl;
    throw DetLayerException("compatibleLayers() method used without all reachableLayers are set"); 
  }

  vector<const DetLayer*> result;

  if ( dir == alongMomentum ) {
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
  if( !areAllReachableLayersSet ){
    cout << "ERROR: compatibleLayers() method used without all reachableLayers are set" << endl;
    throw DetLayerException("compatibleLayers() method used without all reachableLayers are set"); 
  }

  vector<const DetLayer*> result;
  FreeTrajectoryState ftsWithoutErrors = (fts.hasError()) ?
    FreeTrajectoryState(fts.parameters()) : fts;

  if ( dir == alongMomentum ) {
    wellInside(ftsWithoutErrors, dir, theAllOuterLayers, result);
  }
  else { // oppositeToMomentum
    wellInside(ftsWithoutErrors, dir, theAllInnerLayers, result);
  }

  return result;
}


DetLayer* SimpleForwardNavigableLayer::detLayer() const { return theDetLayer; }

void SimpleForwardNavigableLayer::setDetLayer( DetLayer* dl) {
  cerr << "Warning: SimpleForwardNavigableLayer::setDetLayer called."
       << endl << "This should never happen!" << endl;
}

void SimpleForwardNavigableLayer::setInwardLinks(const BDLC& innerBL, 
                                                 const FDLC& innerFL) {

  theInnerBarrelLayers  = innerBL;
  theInnerForwardLayers = innerFL;

  theInnerLayers.reserve(innerBL.size() + innerFL.size());
  for ( ConstBDLI bl = innerBL.begin(); bl != innerBL.end(); bl++ )
    theInnerLayers.push_back(*bl);
  for ( ConstFDLI fl = innerFL.begin(); fl != innerFL.end(); fl++ )
    theInnerLayers.push_back(*fl);

  // sort the inner layers
  sort(theInnerLayers.begin(), theInnerLayers.end(),TkLayerLess(oppositeToMomentum));

}
