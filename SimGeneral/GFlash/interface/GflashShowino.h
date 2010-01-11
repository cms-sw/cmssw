#ifndef GflashShowino_H
#define GflashShowino_H

#include "SimGeneral/GFlash/interface/Gflash3Vector.h"
#include "SimGeneral/GFlash/interface/GflashNameSpace.h"
#include "SimGeneral/GFlash/interface/GflashTrajectory.h"

class GflashShowino {

public:
  //-------------------------
  // Constructor, destructor
  //-------------------------
  GflashShowino();
  ~GflashShowino();

  void initialize(int showerType, double energy, double globalTime,double charge, 
		  Gflash3Vector &position,Gflash3Vector &momentum,
		  double magneticField);

  void updateShowino(double deltaStep);

  int getShowerType() { return theShowerType; }
  double getEnergy() { return theEnergy; }
  double getPathLengthAtShower() { return thePathLengthAtShower; }
  Gflash3Vector& getPositionAtShower() { return thePositionAtShower; }
  double getStepLengthToOut() { return theStepLengthToOut; }
  double getStepLengthToHcal() { return theStepLengthToHcal; }
  GflashTrajectory* getHelix() { return theHelix; }

  double getGlobalTime() { return theGlobalTime; }
  double getPathLength() { return thePathLength; }
  Gflash3Vector& getPosition() { return thePosition; }
  double getEnergyDeposited() { return theEnergyDeposited ; }
  double getDepthAtShower() { return theDepthAtShower ; }

  void setGlobalTime(double globalTime) { theGlobalTime = globalTime; }
  void setPathLength(double pathLength) { thePathLength = pathLength; }
  void setPosition(Gflash3Vector position) { thePosition = position; }
  void addEnergyDeposited(double energy ) { theEnergyDeposited += energy; }

private:
  void evaluateLengths();

private:

  //fixed at the shower starting point
  int theShowerType ; 
  double theEnergy;
  double thePathLengthAtShower;
  Gflash3Vector thePositionAtShower;
  double theStepLengthToOut;
  double theStepLengthToHcal;
  GflashTrajectory* theHelix;

  //updated along the showino trajectory line
  double thePathLength;
  double theGlobalTime;
  Gflash3Vector thePosition;
  double theEnergyDeposited;

  double theDepthAtShower;

};

#endif


