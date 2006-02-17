#include "RecoTracker/TkNavigation/interface/SimpleForwardNavigableLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "Geometry/Surface/interface/BoundCylinder.h"
#include "Geometry/Surface/interface/BoundDisk.h"
#include "RecoTracker/TkNavigation/interface/TkLayerLess.h"

SimpleForwardNavigableLayer::
SimpleForwardNavigableLayer( ForwardDetLayer* detLayer,
			     const BDLC& outerBL, 
			     const FDLC& outerFL, 
			     const MagneticField* field,
			     float epsilon) :
  SimpleNavigableLayer(field,epsilon),
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
