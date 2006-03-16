#include "RecoTracker/TkNavigation/interface/SimpleBarrelNavigableLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "Geometry/Surface/interface/BoundCylinder.h"
#include "Geometry/Surface/interface/BoundDisk.h"
#include "RecoTracker/TkNavigation/interface/TkLayerLess.h"
#include "TrackingTools/DetLayers/src/DetBelowZ.h"

#include <functional>
#include <algorithm>
#include <map>
#include <cmath>

SimpleBarrelNavigableLayer::
SimpleBarrelNavigableLayer( BarrelDetLayer* detLayer,
			    const BDLC& outerBLC, 
			    const FDLC& outerLeftFL, 
			    const FDLC& outerRightFL,
			    const MagneticField* field,
			    float epsilon) :
  SimpleNavigableLayer(field,epsilon),
  theDetLayer( detLayer), 
  theOuterBarrelLayers( outerBLC),
  theOuterLeftForwardLayers( outerLeftFL),
  theOuterRightForwardLayers( outerRightFL)

{
  // put barrel and forward layers together
  theNegOuterLayers.reserve( outerBLC.size() + outerLeftFL.size());
  thePosOuterLayers.reserve( outerBLC.size() + outerRightFL.size());

  for (ConstBDLI bl=outerBLC.begin(); bl!=outerBLC.end(); bl++) 
    theNegOuterLayers.push_back( *bl);
  thePosOuterLayers = theNegOuterLayers; // barrel part the same

  for (ConstFDLI fl=outerLeftFL.begin(); fl!=outerLeftFL.end(); fl++) 
    theNegOuterLayers.push_back( *fl);
  for (ConstFDLI fl=outerRightFL.begin(); fl!=outerRightFL.end(); fl++) 
    thePosOuterLayers.push_back( *fl);

  // sort the outer layers 
  sort( theNegOuterLayers.begin(), theNegOuterLayers.end(), TkLayerLess());
  sort( thePosOuterLayers.begin(), thePosOuterLayers.end(), TkLayerLess());
}
  
vector<const DetLayer*> 
SimpleBarrelNavigableLayer::nextLayers( PropagationDirection dir) const
{
  vector<const DetLayer*> result;
  
  // the order is the one in which layers
  // should be checked for a reasonable trajectory

  if ( dir == alongMomentum) {
    result = theNegOuterLayers;
    for ( DLC::const_iterator i=thePosOuterLayers.begin();
	  i!=thePosOuterLayers.end(); i++) {
      // avoid duplication of barrel layers
      if ((**i).part() == forward) result.push_back(*i);
    }
  }
  else {
    result = theNegInnerLayers;
    for ( DLC::const_iterator i=thePosInnerLayers.begin();
	  i!=thePosInnerLayers.end(); i++) {
      // avoid duplication of barrel layers
      if ((**i).part() == forward) result.push_back(*i);
    }
//     pushResult( result, theInnerBarrelLayers);
//     pushResult( result, theInnerLeftForwardLayers);
//     pushResult( result, theInnerRightForwardLayers);
  }
  return result;
}

vector<const DetLayer*>
SimpleBarrelNavigableLayer::nextLayers( const FreeTrajectoryState& fts, 
					PropagationDirection dir) const
{
  // This method contains the sequence in which the layers are tested.
  // The iteration stops as soon as a layer contains the propagated state
  // within epsilon.

  vector<const DetLayer*> result;

  FreeTrajectoryState ftsWithoutErrors = (fts.hasError()) ?
    FreeTrajectoryState( fts.parameters()) :
    fts;


  //  const BDLC& blc = barrelLayers(fts,dir);

  if ( dir == alongMomentum) {

    if ( fts.momentum().z() > 0) {
      wellInside( ftsWithoutErrors, dir, thePosOuterLayers, result);
    }
    else {
      wellInside( ftsWithoutErrors, dir, theNegOuterLayers, result);
    }
  } 
  else { // oppositeToMomentum
    if ( fts.momentum().z() > 0) {
      wellInside( ftsWithoutErrors, dir, thePosInnerLayers, result);
    }
    else {
      wellInside( ftsWithoutErrors, dir, theNegInnerLayers, result);
    }
  } 

//     // first check the first barrel layer
//     if ( !blc.empty() &&
// 	 wellInside( ftsWithoutErrors, dir, blc.front(), result)) return result;

//     // then all the forward layers
//     const FDLC& flc = forwardLayers( ftsWithoutErrors, dir);
//     if ( wellInside( ftsWithoutErrors, dir, flc.begin(), 
// 		     flc.end(), result)) return result;

//     // then the rest of the barrel layers
//     if ( blc.size() > 1 &&  wellInside( ftsWithoutErrors, dir, blc.begin() + 1,
// 					blc.end(), result)) return result;
//   }

  return result;
}

#include "Utilities/General/interface/CMSexception.h"

vector<const DetLayer*> 
SimpleBarrelNavigableLayer::compatibleLayers( PropagationDirection timeDirection) const
{
  cout << "ERROR: SimpleBarrelNavigableLayer::compatibleLayers() method is not implemented"
       << endl;
  throw(Genexception("ERROR: SimpleBarrelNavigableLayer::compatibleLayers() method is not implemented") );
}

vector<const DetLayer*> 
SimpleBarrelNavigableLayer::compatibleLayers( const FreeTrajectoryState& fts, 
					      PropagationDirection timeDirection) const
{
  cout << "ERROR: SimpleBarrelNavigableLayer::compatibleLayers() method is not implemented"
       << endl;
  throw(Genexception("ERROR: SimpleBarrelNavigableLayer::compatibleLayers() method is not implemented") );
}



const SimpleBarrelNavigableLayer::BDLC&
SimpleBarrelNavigableLayer::barrelLayers( const FreeTrajectoryState& fts,
					  PropagationDirection dir) const
{
  // does not work for momenta pointing inside
  if ( dir == alongMomentum) return theOuterBarrelLayers;
  else                       return theInnerBarrelLayers;
}

const SimpleBarrelNavigableLayer::FDLC&
SimpleBarrelNavigableLayer::forwardLayers( const FreeTrajectoryState& fts,
					   PropagationDirection dir) const
{
  if ( dir == alongMomentum) {
    if ( fts.momentum().z() < 0) {
      return theOuterLeftForwardLayers;
    }
    else {
      return theOuterRightForwardLayers;
    }
  }
  else {
    if ( fts.momentum().z() < 0) {
      return theInnerLeftForwardLayers;
    }
    else {
      return theInnerRightForwardLayers;
    }
  }
}

DetLayer* SimpleBarrelNavigableLayer::detLayer() const { return theDetLayer;}

void   SimpleBarrelNavigableLayer::setDetLayer( DetLayer* dl) {
  cerr << "Warniong: SimpleBarrelNavigableLayer::setDetLayer called."
       << endl << "This should never happen!" << endl;
}

void SimpleBarrelNavigableLayer::setInwardLinks(const BDLC& theBarrelv, 
						const FDLC& theForwardv)
{
  theInnerBarrelLayers=theBarrelv;
  // sort the inner layers
  sort(theInnerBarrelLayers.begin(), theInnerBarrelLayers.end(),TkLayerLess(oppositeToMomentum));


  ConstFDLI middle = find_if( theForwardv.begin(),theForwardv.end(),
			      not1(DetBelowZ(0)));
  theInnerLeftForwardLayers=FDLC(theForwardv.begin(),middle);
  theInnerRightForwardLayers=FDLC(middle,theForwardv.end());

  // sort the inner layers
  sort(theInnerLeftForwardLayers.begin(), theInnerLeftForwardLayers.end(),TkLayerLess(oppositeToMomentum));
  sort(theInnerRightForwardLayers.begin(), theInnerRightForwardLayers.end(),TkLayerLess(oppositeToMomentum));



  // put barrel and forward layers together
  theNegInnerLayers.reserve( theInnerBarrelLayers.size() + theInnerLeftForwardLayers.size());
  thePosInnerLayers.reserve( theInnerBarrelLayers.size() + theInnerRightForwardLayers.size());

  for (ConstBDLI bl=theInnerBarrelLayers.begin(); bl!=theInnerBarrelLayers.end(); bl++) 
    theNegInnerLayers.push_back( *bl);
  thePosInnerLayers = theNegInnerLayers; // barrel part the same

  for (ConstFDLI fl=theInnerLeftForwardLayers.begin(); fl!=theInnerLeftForwardLayers.end(); fl++) 
    theNegInnerLayers.push_back( *fl);
  for (ConstFDLI fl=theInnerRightForwardLayers.begin(); fl!=theInnerRightForwardLayers.end(); fl++) 
    thePosInnerLayers.push_back( *fl);

  // sort the inner layers 
  sort( theNegInnerLayers.begin(), theNegInnerLayers.end(), TkLayerLess(oppositeToMomentum));
  sort( thePosInnerLayers.begin(), thePosInnerLayers.end(), TkLayerLess(oppositeToMomentum));

}
