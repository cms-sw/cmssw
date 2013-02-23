#include "RecoTracker/TkNavigation/interface/SymmetricLayerFinder.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "TrackingTools/DetLayers/src/DetBelowZ.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <functional>
#include <algorithm>
#include <cmath>

using namespace std;

class ForwardLayerMirrorImage : 
  public unary_function< const ForwardDetLayer*, bool> {
public:

  ForwardLayerMirrorImage( const ForwardDetLayer* layer) : theLayer(layer) {}

  bool operator()( const ForwardDetLayer* a) {
    float zdiff = a->position().z() + theLayer->position().z();
    float rdiff = a->specificSurface().innerRadius() - 
      theLayer->specificSurface().innerRadius();

    // equality based on z position and inner radius
    if ( fabs( zdiff) < 1. && fabs( rdiff) < 1.) return true; // units are cm
    else return false;
  }

private:
  const ForwardDetLayer* theLayer;
};

SymmetricLayerFinder::SymmetricLayerFinder( const FDLC& flc)
{
  ConstFDLI middle = find_if( flc.begin(), flc.end(), not1(DetBelowZ(0)));

  FDLC leftLayers = FDLC( flc.begin(),  middle);
  FDLC rightLayers = FDLC( middle, flc.end());
  vector<PairType> foundPairs;

  for ( FDLI i = leftLayers.begin(); i != leftLayers.end(); i++) {
    ForwardDetLayer* partner = mirrorPartner( *i, rightLayers);
    //if ( partner == 0) throw DetLogicError("Assymmetric forward layers in Tracker");
    if ( partner == 0) throw cms::Exception("SymmetricLayerFinder", "Assymmetric forward layers in Tracker");

    foundPairs.push_back( make_pair( *i, partner));
  }

  // fill the map
  for ( vector<PairType>::iterator ipair = foundPairs.begin();
	ipair != foundPairs.end(); ipair++) {
    theForwardMap[ipair->first]  = ipair->second;
    theForwardMap[ipair->second] = ipair->first;
  }
}

ForwardDetLayer* SymmetricLayerFinder::mirrorPartner( const ForwardDetLayer* layer,
						      const FDLC& rightLayers)
{
  ConstFDLI result =
    find_if( rightLayers.begin(), rightLayers.end(), ForwardLayerMirrorImage(layer));
  if ( result == rightLayers.end()) return 0;
  else return *result;
}

SymmetricLayerFinder::FDLC 
SymmetricLayerFinder::mirror( const FDLC& input) {
  FDLC result;
  for ( ConstFDLI  i = input.begin(); i != input.end(); i++) {
    result.push_back( mirror(*i));
  }
  return result;
}
