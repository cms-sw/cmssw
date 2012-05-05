#ifndef HELIXARBITRARYPLANECROSSING2ORDER_H_
#define HELIXARBITRARYPLANECROSSING2ORDER_H_
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/GeomPropagators/interface/HelixPlaneCrossing.h"
#include "FWCore/Utilities/interface/Visibility.h"

/** Calculates intersections of a helix with planes of
 *  any orientation using a parabolic approximation. */

class HelixArbitraryPlaneCrossing2Order GCC11_FINAL : public HelixPlaneCrossing {


public:
  /** Constructor using point, direction and (transverse!) curvature.
   */
  HelixArbitraryPlaneCrossing2Order(const PositionType& point,
				    const DirectionType& direction,
				    const float curvature,
				    const PropagationDirection propDir = alongMomentum);
  /** Fast constructor (for use by HelixArbitraryPlaneCrossing).
   */
  HelixArbitraryPlaneCrossing2Order(const double& x0, const double& y0, const double& z0,
				    const double& cosPhi0, const double& sinPhi0,
				    const double& cosTheta, const double& sinTheta,
				    const double& rho,
				    const PropagationDirection propDir = alongMomentum) :
    theX0(x0), theY0(y0), theZ0(z0),
    theCosPhi0(cosPhi0), theSinPhi0(sinPhi0),
    theCosTheta(cosTheta), theSinThetaI(1./sinTheta),
    theRho(rho), 
    thePropDir(propDir) {}

  // destructor
  virtual ~HelixArbitraryPlaneCrossing2Order() {}

  /** Propagation status (true if valid) and (signed) path length 
   *  along the helix from the starting point to the plane. The 
   *  starting point is given in the constructor.
   */
  virtual std::pair<bool,double> pathLength(const Plane&);

  /** Position at pathlength s from the starting point.
   */
  virtual PositionType position(double s) const;

  /** Direction at pathlength s from the starting point.
   */
  virtual DirectionType direction(double s) const;
  //
  // double precision vectors
  //
  typedef Basic3DVector<double>  PositionTypeDouble;
  typedef Basic3DVector<double>  DirectionTypeDouble;

  /** Position at pathlength s from the starting point in double precision.
   */
  PositionTypeDouble positionInDouble(double s) const;

  /** Direction at pathlength s from the starting point in double precision.
   */
  DirectionTypeDouble directionInDouble(double s) const;

  /** Pathlength to closest solution.
   */
  inline double smallestPathLength (const double firstPathLength,
				    const double secondPathLength) const {
    return fabs(firstPathLength)<fabs(secondPathLength) ? firstPathLength : secondPathLength;
  }

private:

  /** Choice of one of two solutions according to the propagation direction.
   */
  std::pair<bool,double> solutionByDirection(const double dS1,const double dS2) const dso_internal;

private:
  const double theX0,theY0,theZ0;
  double theCosPhi0,theSinPhi0;
  double theCosTheta,theSinThetaI;
  const double theRho;
  const PropagationDirection thePropDir;

};

#endif


