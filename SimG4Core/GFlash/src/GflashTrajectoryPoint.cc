#include "SimG4Core/GFlash/interface/GflashTrajectoryPoint.h"


GflashTrajectoryPoint::GflashTrajectoryPoint() 
  : thePosition(0,0,0), theMomentum(0,0,0), thePathLength(0) 
{
  //default constructor
}

GflashTrajectoryPoint::GflashTrajectoryPoint(G4ThreeVector& position, G4ThreeVector& momentum, 
					     double pathLength)
{
  thePosition = position;
  theMomentum = momentum;
  thePathLength = pathLength;
}

GflashTrajectoryPoint::~GflashTrajectoryPoint() 
{
}
