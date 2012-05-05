#ifndef IterativeHelixExtrapolatorToLine_h_
#define IterativeHelixExtrapolatorToLine_h_

#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/GeomPropagators/interface/HelixLineExtrapolation.h"
#include "TrackingTools/GeomPropagators/interface/HelixExtrapolatorToLine2Order.h"
#include "FWCore/Utilities/interface/Visibility.h"

/** Calculates closest approach of a helix to a line or a point by 
 *  iterative use of a 2nd order expansion of the helix.
 */

class IterativeHelixExtrapolatorToLine GCC11_FINAL : public HelixLineExtrapolation {
public:
  /** Constructor using point, direction and (transverse!) curvature.
   */
  IterativeHelixExtrapolatorToLine (const PositionType& point,
				    const DirectionType& direction,
				    const float curvature,
				    const PropagationDirection propDir = anyDirection);
  // destructor
  virtual ~IterativeHelixExtrapolatorToLine() {}

  /** Propagation status (true if valid) and (signed) path length 
   *  along the helix from the starting point to the closest approach. 
   *  to the point. The starting point is given in the constructor.
   */
  virtual std::pair<bool,double> pathLength (const GlobalPoint& point) const;

  /** Propagation status (true if valid) and (signed) path length 
   *  along the helix from the starting point to the closest approach
   *  to the line. The starting point is given in the constructor.
   */
  virtual std::pair<bool,double> pathLength (const Line& line) const;

  /** Position at pathlength s from the starting point.
   */
  virtual PositionType position (double s) const;

  /** Direction at pathlength s from the starting point.
   */
  virtual DirectionType direction (double s) const;

  /** Position at pathlength s from the starting point.
   */
  PositionTypeDouble positionInDouble (double s) const;

  /** Direction at pathlength s from the starting point.
   */
  DirectionTypeDouble directionInDouble (double s) const;

private:
  /// common functionality for extrapolation to line or point
  template <class T> 
  std::pair<bool,double> genericPathLength (const T& object) const dso_internal;

private:
  const double theX0,theY0,theZ0;
  double theCosPhi0,theSinPhi0;
  double theCosTheta,theSinTheta;
  const double theRho;

  HelixExtrapolatorToLine2Order theQuadraticSolutionFromStart;

  const PropagationDirection thePropDir;

  mutable double theCachedS;
  mutable double theCachedDPhi;
  mutable double theCachedSDPhi;
  mutable double theCachedCDPhi;
};

#endif
