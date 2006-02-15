#ifndef HelixBarrelCylinderCrossing_H
#define HelixBarrelCylinderCrossing_H

#include "Geometry/Vector/interface/Basic2DVector.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/GlobalVector.h"
#include "TrackingTools/GeomPropagators/interface/PropagationDirection.h"

class Cylinder;

/** Calculates the crossing of a helix with a barrel cylinder.
 */ 

class HelixBarrelCylinderCrossing {

  typedef double                   TmpType;
  typedef Basic2DVector<TmpType>   Point; // for private use only
  typedef Basic2DVector<TmpType>   Vector; // for private use only

public:

  typedef GlobalPoint    PositionType;
  typedef GlobalVector   DirectionType;

  HelixBarrelCylinderCrossing( const GlobalPoint& startingPos,
			       const GlobalVector& startingDir,
			       double rho, PropagationDirection propDir, 
			       const Cylinder& cyl);

  bool hasSolution() const { return theSolExists;}

  /** Propagation status (true if valid) and (signed) path length 
   *  along the helix from the starting point to the cylinder. The 
   *  starting point and the cylinder are given in the constructor.
   */
  double pathLength() const { return theS;}

  /** Returns the position along the helix that corresponds to path
   *  length "s" from the starting point. If s is obtained from the
   *  pathLength method the position is the destination point, i.e.
   *  the position of the crossing with a cylinder (if it exists!) 
   *  is given by position( pathLength( cylinder)).
   */
  PositionType position() const { return thePos;}

  /** Returns the direction along the helix that corresponds to path
   *  length "s" from the starting point. As for position,
   *  the direction of the crossing with a cylinder (if it exists!) 
   *  is given by direction( pathLength( cylinder)).
   */
  DirectionType direction() const { return theDir;}

private:

  bool           theSolExists;
  double         theS;
  PositionType   thePos;
  DirectionType  theDir;
  Vector         theD;
  int            theActualDir;

  void chooseSolution( const Point& p1, const Point& p2,
		       const PositionType& startingPos,
		       const DirectionType& startingDir, 
		       PropagationDirection propDir);

};

#endif
