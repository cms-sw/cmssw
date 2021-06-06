#include "TrackingTools/GeomPropagators/interface/StraightLinePlaneCrossing.h"

#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "FWCore/Utilities/interface/isFinite.h"

//
// Propagation status  and path length to intersection
//
std::pair<bool, double> StraightLinePlaneCrossing::pathLength(const Plane& plane) const {
  //
  // calculate path length
  //
  PositionType planePosition(plane.position());
  DirectionType planeNormal(plane.normalVector());
  auto pz = planeNormal.dot(theP0);
  auto dS = -planeNormal.dot(theX0 - planePosition) / pz;
  // check direction
  auto opposite2Track = ((thePropDir == alongMomentum) & (dS < 0.f)) |
                        ((thePropDir == oppositeToMomentum) & (dS > 0.f)) | edm::isNotFinite(dS);
  //
  // Return result
  //
  return std::pair<bool, double>(!opposite2Track, dS);
}

std::pair<bool, StraightLinePlaneCrossing::PositionType> StraightLinePlaneCrossing::position(const Plane& plane) const {
  auto crossed = pathLength(plane);
  if (crossed.first)
    return std::pair<bool, PositionType>(true, position(crossed.second));
  else
    return std::pair<bool, PositionType>(false, PositionType());
}
