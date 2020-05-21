#include "SymmetricLayerFinder.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <algorithm>
#include <cmath>

using namespace std;

SymmetricLayerFinder::SymmetricLayerFinder(const FDLC& flc) {
  ConstFDLI middle =
      find_if(flc.begin(), flc.end(), [](const GeometricSearchDet* a) { return a->position().z() >= 0.0; });

  FDLC leftLayers = FDLC(flc.begin(), middle);
  FDLC rightLayers = FDLC(middle, flc.end());
  vector<PairType> foundPairs;

  for (auto& leftLayer : leftLayers) {
    const ForwardDetLayer* partner = mirrorPartner(leftLayer, rightLayers);
    //if ( partner == 0) throw DetLogicError("Assymmetric forward layers in Tracker");
    if (partner == nullptr)
      throw cms::Exception("SymmetricLayerFinder", "Assymmetric forward layers in Tracker");

    foundPairs.push_back(make_pair(leftLayer, partner));
  }

  // fill the map
  for (auto& foundPair : foundPairs) {
    theForwardMap[foundPair.first] = foundPair.second;
    theForwardMap[foundPair.second] = foundPair.first;
  }
}

const ForwardDetLayer* SymmetricLayerFinder::mirrorPartner(const ForwardDetLayer* layer, const FDLC& rightLayers) {
  auto mirrorImage = [=](const ForwardDetLayer* a) -> bool {
    auto zdiff = a->position().z() + layer->position().z();
    auto rdiff = a->specificSurface().innerRadius() - layer->specificSurface().innerRadius();

    // equality based on z position and inner radius
    return std::abs(zdiff) < 2.f && std::abs(rdiff) < 1.f;  // units are cm
  };

  ConstFDLI result = find_if(rightLayers.begin(), rightLayers.end(), mirrorImage);
  if (result == rightLayers.end())
    return nullptr;
  else
    return *result;
}

SymmetricLayerFinder::FDLC SymmetricLayerFinder::mirror(const FDLC& input) {
  FDLC result;
  for (auto i : input) {
    result.push_back(mirror(i));
  }
  return result;
}
