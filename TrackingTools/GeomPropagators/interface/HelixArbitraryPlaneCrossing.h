#ifndef HELIXARBITRARYPLANECROSSING_H_
#define HELIXARBITRARYPLANECROSSING_H_
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/GeomPropagators/interface/HelixPlaneCrossing.h"
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing2Order.h"
#include "FWCore/Utilities/interface/Visibility.h"

/** Calculates intersections of a helix with planes of any orientation. */

class HelixArbitraryPlaneCrossing final : public HelixPlaneCrossing {
public:
  /** Constructor using point, direction and (transverse!) curvature.
   */
  HelixArbitraryPlaneCrossing(const PositionType& point,
				    const DirectionType& direction,
				    const float curvature,
			            const PropagationDirection propDir = alongMomentum);
  // destructor
  ~HelixArbitraryPlaneCrossing() override {}

  /** Propagation status (true if valid) and (signed) path length 
   *  along the helix from the starting point to the plane. The 
   *  starting point is given in the constructor.
   */
  std::pair<bool,double> pathLength(const Plane& plane) override;

  /** Position at pathlength s from the starting point.
   */
  PositionType position(double s) const override;

  /** Direction at pathlength s from the starting point.
   */
  DirectionType direction(double s) const override;
  //
  // double precision vectors for internal use
  //
  typedef Basic3DVector<double>  PositionTypeDouble;
  typedef Basic3DVector<double>  DirectionTypeDouble;

  /** Position at pathlength s from the starting point.
   */
  PositionTypeDouble positionInDouble(double s) const;

  /** Direction at pathlength s from the starting point.
   */
  DirectionTypeDouble directionInDouble(double s) const;

private:
  /** Iteration control: check for significant distance to plane.
   */
  inline bool notAtSurface (const Plane&,
  			    const PositionTypeDouble&,
			    const float) const dso_internal;

private:
  HelixArbitraryPlaneCrossing2Order theQuadraticCrossingFromStart;


  const double theX0,theY0,theZ0;
  double theCosPhi0,theSinPhi0;
  double theCosTheta,theSinTheta;
  const double theRho;

  const PropagationDirection thePropDir;

  mutable double theCachedS;
  mutable double theCachedDPhi;
  mutable double theCachedSDPhi;
  mutable double theCachedCDPhi;

  static const float theNumericalPrecision;
  static const float theMaxDistToPlane;

};

#endif
