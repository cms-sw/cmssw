#ifndef SimG4Core_SteppingTrackStatus_H
#define SimG4Core_SteppingTrackStatus_H

enum TrackStatus {
  sAlive = 0,
  sKilledByProcess = 1,
  sDeadRegion = 2,
  sOutOfTime = 3,
  sLowEnergy = 4,
  sLowEnergyInVacuum = 5,
  sEnergyDepNaN = 6,
  sVeryForward = 7,
  sNumberOfSteps = 8
};

#endif
