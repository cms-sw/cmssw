#include "SimGeneral/GFlash/interface/GflashShowino.h"

GflashShowino::GflashShowino() 
  : theShowerType(-1), theEnergy(0), thePathLengthAtShower(0), thePositionAtShower(0,0,0), 
    theStepLengthToOut(0), theStepLengthToHcal(0), thePathLength(0), theGlobalTime(0), 
    thePosition(0,0,0), theEnergyDeposited(0), theDepthAtShower(0)
{
  theHelix =  new GflashTrajectory;;
}

GflashShowino::~GflashShowino() 
{
  delete theHelix;
}

void GflashShowino::initialize(int showerType, double energy, double globalTime, double charge, 
			       Gflash3Vector &position, Gflash3Vector &momentum,
			       double magneticField) {
  theShowerType = showerType;
  theEnergy = energy;
  theGlobalTime = globalTime;
  thePositionAtShower = position;  
  thePosition = thePositionAtShower;
  
  // inside the magnetic field (tesla unit);
  theHelix->initializeTrajectory(momentum,thePosition,charge,magneticField);

  evaluateLengths();

  theEnergyDeposited = 0.0;

}

void GflashShowino::updateShowino(double deltaStep)
{
  thePathLength += deltaStep;
  //trajectory point of showino along the shower depth at the pathLength
  GflashTrajectoryPoint trajectoryShowino;
  theHelix->getGflashTrajectoryPoint(trajectoryShowino,thePathLength);

  thePosition = trajectoryShowino.getPosition();

  theGlobalTime +=  (theEnergy/100.0)*deltaStep/30.0; //@@@calculate exact time change in nsec
  //  theGlobalTime +=  (theEnergy/100.0)*deltaStep/3.0000e+10; //@@@calculate exact time change in nsec
}

void GflashShowino::evaluateLengths() {
  Gflash::CalorimeterNumber kCalor = Gflash::getCalorimeterNumber(thePosition);

  //thePathLengthAtShower: path Length from the origin to the shower starting point in cm
  //theStepLengthToOut: the total path length from the starting point of
  //                    shower to the maximum distance inside paramerized envelopes
  
  if(kCalor == Gflash::kESPM || kCalor == Gflash::kHB ) {
    thePathLengthAtShower = theHelix->getPathLengthAtRhoEquals(thePosition.getRho());
    theStepLengthToOut  = theHelix->getPathLengthAtRhoEquals(Gflash::Rmax[Gflash::kHB]) - thePathLengthAtShower;
    theStepLengthToHcal = theHelix->getPathLengthAtRhoEquals(Gflash::Rmin[Gflash::kHB]) - thePathLengthAtShower;
  }
  else if (kCalor == Gflash::kENCA || kCalor == Gflash::kHE ) {
    thePathLengthAtShower = theHelix->getPathLengthAtZ(thePosition.getZ());
    theStepLengthToOut  = theHelix->getPathLengthAtZ(Gflash::Zmax[Gflash::kHE]) - thePathLengthAtShower;
    theStepLengthToHcal = theHelix->getPathLengthAtZ(Gflash::Zmin[Gflash::kHE]) - thePathLengthAtShower;
  }
  else { 
    //@@@extend for HF later
    theStepLengthToOut = 200.0;
  }

  thePathLength = thePathLengthAtShower;

}

