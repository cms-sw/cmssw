#ifndef GflashTrajectoryPoint_H
#define GflashTrajectoryPoint_H

#include "G4ThreeVector.hh"

class GflashTrajectoryPoint {

public:
  //-------------------------
  // Constructor, destructor
  //-------------------------
  GflashTrajectoryPoint();

  GflashTrajectoryPoint(G4ThreeVector& position, G4ThreeVector& momentum, 
			double pathLength);

  ~GflashTrajectoryPoint();

  double getPathLength() { return thePathLength; }
  G4ThreeVector& getPosition() { return thePosition; }
  G4ThreeVector& getMomentum() { return theMomentum; }
  G4ThreeVector  getOrthogonalUnitVector() { return theMomentum.orthogonal().unit(); }
  G4ThreeVector  getCrossUnitVector() { return theMomentum.cross(getOrthogonalUnitVector()).unit(); }

  void setPosition(const G4ThreeVector& position ) { thePosition = position; }
  void setMomentum(const G4ThreeVector& momentum ) { theMomentum = momentum; }
  void setPathLength(double pathLength ) { thePathLength = pathLength; }

private:
  G4ThreeVector thePosition;
  G4ThreeVector theMomentum;
  double thePathLength;
};

#endif


