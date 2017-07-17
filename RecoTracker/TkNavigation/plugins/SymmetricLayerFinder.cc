#include "SymmetricLayerFinder.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "TrackingTools/DetLayers/src/DetBelowZ.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <functional>
#include <algorithm>
#include <cmath>

using namespace std;


SymmetricLayerFinder::SymmetricLayerFinder( const FDLC& flc)
{
  ConstFDLI middle = find_if( flc.begin(), flc.end(), not1(DetBelowZ(0)));

  FDLC leftLayers = FDLC( flc.begin(),  middle);
  FDLC rightLayers = FDLC( middle, flc.end());
  vector<PairType> foundPairs;

  for ( FDLI i = leftLayers.begin(); i != leftLayers.end(); i++) {
   const  ForwardDetLayer* partner = mirrorPartner( *i, rightLayers);
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

const ForwardDetLayer* SymmetricLayerFinder::mirrorPartner( const ForwardDetLayer* layer,
						      const FDLC& rightLayers)
{

  auto mirrorImage = [=]( const ForwardDetLayer* a) ->bool {
    auto zdiff = a->position().z() + layer->position().z();
    auto rdiff = a->specificSurface().innerRadius() -
      layer->specificSurface().innerRadius();

    // equality based on z position and inner radius
    return std::abs(zdiff) < 2.f && std::abs(rdiff) < 1.f; // units are cm
  };


  ConstFDLI result =
    find_if( rightLayers.begin(), rightLayers.end(), mirrorImage);
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
