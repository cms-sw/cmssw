#ifndef HelixExtrapolatorToLine2Order_h_
#define HelixExtrapolatorToLine2Order_h_

#include "TrackingTools/GeomPropagators/interface/HelixLineExtrapolation.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "FWCore/Utilities/interface/Visibility.h"

/** Calculates intersections of a helix with planes of
 *  any orientation using a parabolic approximation. */

class HelixExtrapolatorToLine2Order GCC11_FINAL : public HelixLineExtrapolation {
public:
  /// Constructor using point, direction and (transverse!) curvature.
  HelixExtrapolatorToLine2Order(const PositionType& point,
				const DirectionType& direction,
				const float curvature,
				const PropagationDirection propDir = alongMomentum);

  /// Fast constructor (for use by IterativeHelixExtrapolatorToLine).
  HelixExtrapolatorToLine2Order(const double& x0, const double& y0, const double& z0,
				const double& cosPhi0, const double& sinPhi0,
				const double& cosTheta, const double& sinTheta,
				const double& rho,
				const PropagationDirection propDir = alongMomentum) :
    thePosition(x0,y0,z0),
    theDirection(cosPhi0,sinPhi0,cosTheta/sinTheta),
    theSinTheta(sinTheta),
    theRho(rho), 
    thePropDir(propDir) {}
  
  // destructor
  virtual ~HelixExtrapolatorToLine2Order() {}

  /** Propagation status (true if valid) and (signed) path length 
   *  along the helix from the starting point to the closest approach
   *  to the point. The starting point is given in the constructor.
   */
  virtual std::pair<bool,double> pathLength (const GlobalPoint& point) const;

  /** Propagation status (true if valid) and (signed) path length 
   *  along the helix from the starting point to the closest approach
   *  to the line. The starting point is given in the constructor.
   */
  virtual std::pair<bool,double> pathLength (const Line& line) const;

  /// Position at pathlength s from the starting point.
  virtual PositionType position(double s) const;

  /// Direction at pathlength s from the starting point.
  virtual DirectionType direction(double s) const;

  /// Position at pathlength s from the starting point in double precision.
  PositionTypeDouble positionInDouble(double s) const;

  /// Direction at pathlength s from the starting point in double precision.
  DirectionTypeDouble directionInDouble(double s) const;

private:
  /// common part for propagation to point and line
  virtual std::pair<bool,double> pathLengthFromCoefficients (const double ceq[4]) const dso_internal;
  /// Solutions of 3rd order equation
  int solve3rdOrder (const double ceq[], double sol[]) const dso_internal;
  /// Solutions of 2nd order equation
  int solve2ndOrder (const double ceq[], double sol[]) const dso_internal;

private:
  const PositionTypeDouble thePosition;
  DirectionTypeDouble theDirection;
  double theSinTheta;
  const double theRho;
  const PropagationDirection thePropDir;
};

#endif


