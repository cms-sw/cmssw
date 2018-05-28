#ifndef HelixForwardPlaneCrossing_H_
#define HelixForwardPlaneCrossing_H_

#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/GeomPropagators/interface/HelixPlaneCrossing.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "FWCore/Utilities/interface/Likely.h"
#include <limits>

/** Calculates intersections of a helix with planes perpendicular to the z-axis.
 */

class HelixForwardPlaneCrossing final : public HelixPlaneCrossing {
public:
  /** Constructor using point, direction and (transverse!) curvature.
   */
  HelixForwardPlaneCrossing(const PositionType& point,
			    const DirectionType& direction,
			    const float curvature,
			    const PropagationDirection propDir = alongMomentum);
  // destructor
  ~HelixForwardPlaneCrossing() override {}

  /** Propagation status (true if valid) and (signed) path length 
   *  along the helix from the starting point to the plane.
   */
  std::pair<bool,double> pathLength(const Plane& plane) override {
    //
    // Protect against p_z=0 and calculate path length
    //
    if UNLIKELY( std::abs(theCosTheta)<std::numeric_limits<float>::min()  )  return std::pair<bool,double>(false,0);
    
    double dS = (plane.position().z()-theZ0) / theCosTheta;
    
    // negative logic to avoid checking for anyDirection...
    return std::make_pair( ! ( 
			      ( (thePropDir==alongMomentum) & (dS<0.) ) |
			      ( (thePropDir==oppositeToMomentum) & (dS>0.) )  
			       ) , dS);

  }

  /** Position at pathlength s from the starting point.
   */
  PositionType position(double s) const override;

  /** Direction at pathlength s from the starting point.
   */
  DirectionType direction(double s) const override;

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
