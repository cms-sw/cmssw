#ifndef HelixBarrelPlaneCrossingByCircle_H
#define HelixBarrelPlaneCrossingByCircle_H

#include "TrackingTools/GeomPropagators/interface/HelixPlaneCrossing.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

/** Computes the crossing of a helix with a barrel plane.
 *  Exact if the magnetic field is parallel to the plane.
 */

class HelixBarrelPlaneCrossingByCircle GCC11_FINAL : public HelixPlaneCrossing {
public:

  HelixBarrelPlaneCrossingByCircle( const PositionType& pos,
				    const DirectionType& dir,
				    double rho, 
				    PropagationDirection propDir=alongMomentum);

  HelixBarrelPlaneCrossingByCircle( const GlobalPoint& pos,
				    const GlobalVector& dir,
				    double rho, 
				    PropagationDirection propDir=alongMomentum);

  virtual std::pair<bool,double> pathLength( const Plane&);

  virtual PositionType position( double s) const;

  virtual DirectionType direction( double s) const;

private:

  typedef Basic2DVector<double> Vector2D;

  PositionType theStartingPos;
  DirectionType theStartingDir;
  double theRho;
  PropagationDirection thePropDir;

  double theCosTheta;
  double theSinTheta;
  double theXCenter;
  double theYCenter;

  // caching of the solution for faster access
  double   theS;
  Vector2D theD;
  double   theDmag;

  // internal communication - not very clean
  double   theActualDir;
  bool useStraightLine;

  void init();
  bool chooseSolution( const Vector2D& d1, const Vector2D& d2);

};

#endif
