#ifndef HelixBarrelPlaneCrossing2OrderLocal_H
#define HelixBarrelPlaneCrossing2OrderLocal_H

#include "DataFormats/GeometrySurface/interface/Plane.h"

/** Calculates an approximate crossing of a helix and a barrel plane.
 *  The helix circle is approximated with a parabola.
 *  The current class name is misleading, since it does not have the
 *  HelixPlaneCrossing interface.
 */

class HelixBarrelPlaneCrossing2OrderLocal {
public:

  typedef Surface::GlobalPoint    GlobalPoint;
  typedef Surface::GlobalVector   GlobalVector;
  typedef Surface::LocalPoint     LocalPoint;
  typedef Surface::LocalVector    LocalVector;

  HelixBarrelPlaneCrossing2OrderLocal() {}

  HelixBarrelPlaneCrossing2OrderLocal( const GlobalPoint& startingPos,
				       const GlobalVector& startingDir,
				       double rho, 
				       const Plane& plane);

  LocalPoint  position() const { return thePos;}
  LocalVector direction() const { return theDir;}

private:

  typedef Basic2DVector<float>  Vector2D;
 
  LocalPoint  thePos;
  LocalVector theDir;

};

#endif
