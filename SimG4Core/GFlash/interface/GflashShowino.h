#ifndef GflashShowino_H
#define GflashShowino_H

#include "SimG4Core/GFlash/interface/GflashNameSpace.h"
#include "SimG4Core/GFlash/interface/GflashTrajectory.h"
#include "G4FastTrack.hh"
//#include "G4Track.hh"
#include "G4ThreeVector.hh"

class GflashShowino {

public:
  //-------------------------
  // Constructor, destructor
  //-------------------------
  GflashShowino();
  ~GflashShowino();

  void initializeShowino(const G4FastTrack& fastTrack, G4double magneticField);
  void updateShowino(G4double deltaStep);

  G4double getEnergy() { return theEnergy; }
  G4double getPathLengthAtShower() { return thePathLengthAtShower; }
  G4ThreeVector& getPositionAtShower() { return thePositionAtShower; }
  G4double getStepLengthToOut() { return theStepLengthToOut; }
  G4double getStepLengthToHcal() { return theStepLengthToHcal; }
  GflashTrajectory* getHelix() { return theHelix; }

  G4double getGlobalTime() { return theGlobalTime; }
  G4double getPathLength() { return thePathLength; }
  G4ThreeVector& getPosition() { return thePosition; }
  G4double getEnergyDeposited() { return theEnergyDeposited ; }

  void setGlobalTime(G4double globalTime) { theGlobalTime = globalTime; }
  void setPathLength(G4double pathLength) { thePathLength = pathLength; }
  void setPosition(G4ThreeVector position) { thePosition = position; }
  void addEnergyDeposited(G4double energy ) { theEnergyDeposited += energy; }

private:
  void evaluateLengths();

private:
  //  const G4Track* thePrimaryTrack;

  //fixed at the shower starting point
  G4double theEnergy;
  G4double thePathLengthAtShower;
  G4ThreeVector thePositionAtShower;
  G4double theStepLengthToOut;
  G4double theStepLengthToHcal;
  GflashTrajectory* theHelix;

  //updated along the showino trajectory line
  G4double thePathLength;
  G4double theGlobalTime;
  G4ThreeVector thePosition;
  G4double theEnergyDeposited;
};

#endif


