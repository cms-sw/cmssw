#ifndef GflashTrajectoryPoint_H
#define GflashTrajectoryPoint_H

#include "CLHEP/Geometry/Point3D.h"
#include "CLHEP/Geometry/Vector3D.h"
#include "SimGeneral/GFlash/interface/Gflash3Vector.h"

class GflashTrajectoryPoint {
public:
  //-------------------------
  // Constructor, destructor
  //-------------------------
  GflashTrajectoryPoint();

  GflashTrajectoryPoint(Gflash3Vector &position, Gflash3Vector &momentum, double pathLength);

  ~GflashTrajectoryPoint();

  double getPathLength() { return thePathLength; }
  Gflash3Vector &getPosition() { return thePosition; }
  Gflash3Vector &getMomentum() { return theMomentum; }
  Gflash3Vector getOrthogonalUnitVector() { return theMomentum.orthogonal().unit(); }
  Gflash3Vector getCrossUnitVector() { return theMomentum.cross(getOrthogonalUnitVector()).unit(); }

  void setPosition(const Gflash3Vector &position) { thePosition = position; }
  void setMomentum(const Gflash3Vector &momentum) { theMomentum = momentum; }
  void setPathLength(double pathLength) { thePathLength = pathLength; }

private:
  Gflash3Vector thePosition;
  Gflash3Vector theMomentum;
  double thePathLength;
};

#endif
