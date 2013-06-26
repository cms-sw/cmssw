#ifndef HelixForwardPlaneCrossing_H_
#define HelixForwardPlaneCrossing_H_

#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/GeomPropagators/interface/HelixPlaneCrossing.h"

/** Calculates intersections of a helix with planes perpendicular to the z-axis.
 */

class HelixForwardPlaneCrossing GCC11_FINAL : public HelixPlaneCrossing {
public:
  /** Constructor using point, direction and (transverse!) curvature.
   */
  HelixForwardPlaneCrossing(const PositionType& point,
			    const DirectionType& direction,
			    const float curvature,
			    const PropagationDirection propDir = alongMomentum);
  // destructor
  virtual ~HelixForwardPlaneCrossing() {}

  /** Propagation status (true if valid) and (signed) path length 
   *  along the helix from the starting point to the plane.
   */
  virtual std::pair<bool,double> pathLength(const Plane& plane);

  /** Position at pathlength s from the starting point.
   */
  virtual PositionType position(double s) const;

  /** Direction at pathlength s from the starting point.
   */
  virtual DirectionType direction(double s) const;

private:
  //
  // double precision vectors for internal use
  //
  typedef Basic3DVector<double>  PositionTypeDouble;
  typedef Basic3DVector<double>  DirectionTypeDouble;

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
};

#endif
