#include "SimG4Core/GFlash/interface/GflashShowino.h"
#include "SimG4Core/GFlash/interface/GflashTrajectoryPoint.h"
#include "G4TouchableHandle.hh"

GflashShowino::GflashShowino() 
  : theEnergy(0), thePathLengthAtShower(0), thePositionAtShower(0,0,0), theStepLengthToOut(0), 
    theStepLengthToHcal(0), thePathLength(0), theGlobalTime(0), thePosition(0,0,0), theEnergyDeposited(0)
{
  //default constructor
  //  thePrimaryTrack = 0;
  theHelix =  new GflashTrajectory;;
}

GflashShowino::~GflashShowino() 
{
  delete theHelix;
}

void GflashShowino::initializeShowino(const G4FastTrack& fastTrack, G4double magneticField)
{
  theEnergy = fastTrack.GetPrimaryTrack()->GetKineticEnergy()/GeV;
  theGlobalTime = fastTrack.GetPrimaryTrack()->GetStep()->GetPostStepPoint()->GetGlobalTime();
  thePositionAtShower = fastTrack.GetPrimaryTrack()->GetPosition()/cm;

  thePosition = thePositionAtShower;

  // inside the magnetic field (tesla unit);
  G4ThreeVector momentum = fastTrack.GetPrimaryTrack()->GetMomentum()/GeV;
  G4double charge = fastTrack.GetPrimaryTrack()->GetStep()->GetPreStepPoint()->GetCharge();
  theHelix->initializeTrajectory(momentum,thePosition,charge,magneticField);

  evaluateLengths();
  
  theEnergyDeposited = 0.0;
}

void GflashShowino::updateShowino(G4double deltaStep)
{
  thePathLength += deltaStep;
  //trajectory point of showino along the shower depth at the pathLength
  GflashTrajectoryPoint trajectoryShowino;
  theHelix->getGflashTrajectoryPoint(trajectoryShowino,thePathLength);

  thePosition = trajectoryShowino.getPosition();
  theGlobalTime +=  (theEnergy/100.0)*deltaStep/3.0000e+10; //@@@calculate exact time change
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
