#include "RecoTracker/TkNavigation/interface/SimpleBarrelNavigableLayer.h"

#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"

#include "TrackingTools/DetLayers/src/DetBelowZ.h"
#include "TrackingTools/DetLayers/interface/DetLayerException.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"

#include "RecoTracker/TkNavigation/interface/TkLayerLess.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <functional>
#include <algorithm>
#include <map>
#include <cmath>

using namespace std;


SimpleBarrelNavigableLayer::
SimpleBarrelNavigableLayer( BarrelDetLayer* detLayer,
			    const BDLC& outerBLC, 
			    const FDLC& outerLeftFL, 
			    const FDLC& outerRightFL,
			    const MagneticField* field,
			    float epsilon,
			    bool checkCrossingSide) :
  SimpleNavigableLayer(field,epsilon,checkCrossingSide),
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
  sort(theOuterBarrelLayers.begin(), theOuterBarrelLayers.end(), TkLayerLess());
  sort(theOuterLeftForwardLayers.begin(), theOuterLeftForwardLayers.end(), TkLayerLess());
  sort(theOuterRightForwardLayers.begin(), theOuterRightForwardLayers.end(), TkLayerLess());
}
  


vector<const DetLayer*> 
SimpleBarrelNavigableLayer::nextLayers( NavigationDirection dir) const
{
  vector<const DetLayer*> result;
  
  // the order is the one in which layers
  // should be checked for a reasonable trajectory

  if ( dir == insideOut) {
    result = theNegOuterLayers;
    for ( DLC::const_iterator i=thePosOuterLayers.begin();
	  i!=thePosOuterLayers.end(); i++) {
      // avoid duplication of barrel layers
      if ((**i).location() == GeomDetEnumerators::endcap) result.push_back(*i);
    }
  }
  else {
    result = theNegInnerLayers;
    for ( DLC::const_iterator i=thePosInnerLayers.begin();
	  i!=thePosInnerLayers.end(); i++) {
      // avoid duplication of barrel layers
      if ((**i).location() == GeomDetEnumerators::endcap) result.push_back(*i);
    }
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

  //establish whether the tracks is crossing the tracker from outer layers to inner ones 
  //or from inner to outer.
  GlobalVector transversePosition(fts.position().x(), fts.position().y(), 0);
  //GlobalVector transverseMomentum(fts.momentum().x(), fts.momentum().y(), 0);
  //bool isInOutTrack  = (fts.position().basicVector().dot(fts.momentum().basicVector())>0) ? 1 : 0;
  bool isInOutTrackBarrel  = (transversePosition.dot(fts.momentum())>0) ? 1 : 0;
  //cout << "dot: " << transversePosition.dot(fts.momentum()) << endl;

  float zpos = fts.position().z();
  bool isInOutTrackFWD = fts.momentum().z()*zpos>0;
  

  //establish whether inner or outer layers are crossed after propagation, according
  //to BOTH propagationDirection AND track momentum
  bool dirOppositeXORisInOutTrackBarrel = ( !(dir == oppositeToMomentum) && isInOutTrackBarrel) || ((dir == oppositeToMomentum) && !isInOutTrackBarrel);
  bool dirOppositeXORisInOutTrackFWD = ( !(dir == oppositeToMomentum) && isInOutTrackFWD) || ((dir == oppositeToMomentum) && !isInOutTrackFWD);

  LogDebug("SimpleBarrelNavigableLayer") << "is alongMomentum? " << (dir == alongMomentum) << endl
					 << "isInOutTrackBarrel: " << isInOutTrackBarrel << endl
					 << "isInOutTrackFWD: " << isInOutTrackFWD << endl
					 << "dirOppositeXORisInOutTrackFWD: " << dirOppositeXORisInOutTrackFWD << endl
					 << "dirOppositeXORisInOutTrackBarrel: "<< dirOppositeXORisInOutTrackBarrel << endl;

  bool signZmomentumXORdir = (( (fts.momentum().z() > 0) && !(dir == alongMomentum) ) ||
			      (!(fts.momentum().z() > 0) &&  (dir == alongMomentum) )   );


  if ( dirOppositeXORisInOutTrackBarrel &&  dirOppositeXORisInOutTrackFWD) {

    if ( signZmomentumXORdir   ) {
      wellInside( ftsWithoutErrors, dir, theNegOuterLayers, result);
    }
    else {
      wellInside( ftsWithoutErrors, dir, thePosOuterLayers, result);
    }
  } else if (!dirOppositeXORisInOutTrackBarrel &&  !dirOppositeXORisInOutTrackFWD){
    if ( signZmomentumXORdir ) {
      wellInside( ftsWithoutErrors, dir, thePosInnerLayers, result);
    }
    else {
      wellInside( ftsWithoutErrors, dir, theNegInnerLayers, result);
    }
  } else if (!dirOppositeXORisInOutTrackBarrel && dirOppositeXORisInOutTrackFWD){
    wellInside(ftsWithoutErrors, dir, theInnerBarrelLayers.begin(), theInnerBarrelLayers.end(), result);	

    if (signZmomentumXORdir){	
      wellInside(ftsWithoutErrors, dir, theInnerLeftForwardLayers.begin(), theInnerLeftForwardLayers.end(), result);
      wellInside(ftsWithoutErrors, dir, theOuterLeftForwardLayers.begin(), theOuterLeftForwardLayers.end(), result);	
    }	else {
      wellInside(ftsWithoutErrors, dir, theInnerRightForwardLayers.begin(), theInnerRightForwardLayers.end(), result);
      wellInside(ftsWithoutErrors, dir, theOuterRightForwardLayers.begin(), theOuterRightForwardLayers.end(), result);
    }
  } else {
     if (signZmomentumXORdir){
        wellInside(ftsWithoutErrors, dir, theInnerLeftForwardLayers.begin(), theInnerLeftForwardLayers.end(), result);
     } else {
        wellInside(ftsWithoutErrors, dir, theInnerRightForwardLayers.begin(), theInnerRightForwardLayers.end(), result);
     }	
     wellInside(ftsWithoutErrors, dir, theOuterBarrelLayers.begin(), theOuterBarrelLayers.end(), result);	
  }

  bool goingIntoTheBarrel = (!isInOutTrackBarrel && dir==alongMomentum) || (isInOutTrackBarrel && dir==oppositeToMomentum) ;

  LogDebug("SimpleBarrelNavigableLayer") << "goingIntoTheBarrel: " << goingIntoTheBarrel;


  if unlikely(theSelfSearch && result.size()==0){
    if (!goingIntoTheBarrel){     
      LogDebug("SimpleBarrelNavigableLayer")<<" state is not going toward the center of the barrel. not adding self search.";}
    else{
      const BarrelDetLayer * bl = reinterpret_cast<const BarrelDetLayer *>(detLayer());      unsigned int before=result.size();
      LogDebug("SimpleBarrelNavigableLayer")<<" I am trying to added myself as a next layer.";
      wellInside(ftsWithoutErrors, dir, bl, result);
      unsigned int after=result.size();
      if (before!=after)
	LogDebug("SimpleBarrelNavigableLayer")<<" I have added myself as a next layer.";
    }
  }
  
  return result;
}


vector<const DetLayer*> 
SimpleBarrelNavigableLayer::compatibleLayers( NavigationDirection dir) const {
  edm::LogError("TkNavigation") << "ERROR: compatibleLayers() method used without all reachableLayers are set" ;
  throw DetLayerException("compatibleLayers() method used without all reachableLayers are set"); 
  return vector<const DetLayer*>();
}



void   SimpleBarrelNavigableLayer::setDetLayer( DetLayer* dl) {
  cerr << "Warniong: SimpleBarrelNavigableLayer::setDetLayer called."
       << endl << "This should never happen!" << endl;
}

void SimpleBarrelNavigableLayer::setInwardLinks(const BDLC& theBarrelv, 
						const FDLC& theForwardv,
						TkLayerLess sorter)
{
  theInnerBarrelLayers=theBarrelv;
  // sort the inner layers
  sort(theInnerBarrelLayers.begin(), theInnerBarrelLayers.end(),sorter);


  ConstFDLI middle = find_if( theForwardv.begin(),theForwardv.end(),
			      not1(DetBelowZ(0)));
  theInnerLeftForwardLayers=FDLC(theForwardv.begin(),middle);
  theInnerRightForwardLayers=FDLC(middle,theForwardv.end());

  // sort the inner layers
  sort(theInnerLeftForwardLayers.begin(), theInnerLeftForwardLayers.end(),sorter);
  sort(theInnerRightForwardLayers.begin(), theInnerRightForwardLayers.end(),sorter);



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
  sort( theNegInnerLayers.begin(), theNegInnerLayers.end(), sorter);
  sort( thePosInnerLayers.begin(), thePosInnerLayers.end(), sorter);
  sort(theInnerBarrelLayers.begin(), theInnerBarrelLayers.end(),sorter);
  sort(theInnerLeftForwardLayers.begin(), theInnerLeftForwardLayers.end(),sorter);
  sort(theInnerRightForwardLayers.begin(), theInnerRightForwardLayers.end(),sorter);

}

void SimpleBarrelNavigableLayer::setAdditionalLink(DetLayer* additional, NavigationDirection direction){
  ForwardDetLayer* fadditional = dynamic_cast<ForwardDetLayer*>(additional);
  BarrelDetLayer*  badditional = dynamic_cast<BarrelDetLayer*>(additional);
  if (badditional){	
  	if (direction==insideOut){
		theOuterBarrelLayers.push_back(badditional);
		theNegOuterLayers.push_back(badditional);
		thePosOuterLayers.push_back(badditional);
		return;	
  	}
  	theInnerBarrelLayers.push_back(badditional);
	theNegInnerLayers.push_back(badditional);
        thePosInnerLayers.push_back(badditional);	
	return;
  } else if (fadditional){
	double zpos = fadditional->position().z(); 
	if (direction==insideOut){
		if (zpos>0){
			theOuterRightForwardLayers.push_back(fadditional);
			thePosOuterLayers.push_back(fadditional);
			return;	
		}
		theOuterLeftForwardLayers.push_back(fadditional);
		theNegOuterLayers.push_back(fadditional);
		return;
	}
	if (zpos>0){
        	theInnerRightForwardLayers.push_back(fadditional);
		thePosInnerLayers.push_back(fadditional);
                return;
        }
        theInnerLeftForwardLayers.push_back(fadditional);
	theNegInnerLayers.push_back(fadditional);
        return;
  }
  edm::LogError("TkNavigation") << "trying to add neither a ForwardDetLayer nor a BarrelDetLayer";	
  return;	
}
