#ifndef GflashEMShowerProfile_H
#define GflashEMShowerProfile_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimGeneral/GFlash/interface/GflashNameSpace.h"
#include "SimGeneral/GFlash/interface/GflashTrajectory.h"

#include <vector>

class GflashHit;
class GflashShowino;
class GflashHistogram;

class GflashEMShowerProfile {
public:
  //-------------------------
  // Constructor, destructor
  //-------------------------
  GflashEMShowerProfile(const edm::ParameterSet &parSet);
  ~GflashEMShowerProfile();

  void initialize(int showerType,
                  double energy,
                  double globalTime,
                  double charge,
                  Gflash3Vector &position,
                  Gflash3Vector &momentum);

  void parameterization();
  GflashShowino *getGflashShowino() { return theShowino; }
  std::vector<GflashHit> &getGflashHitList() { return theGflashHitList; };

private:
  double getDistanceToOut(Gflash::CalorimeterNumber kCalor);
  Gflash3Vector locateHitPosition(
      GflashTrajectoryPoint &point, double rCore, double rTail, double probability, double &rShower);

private:
  Gflash::CalorimeterNumber jCalorimeter;

  edm::ParameterSet theParSet;
  double theBField;
  double theEnergyScaleEB;
  double theEnergyScaleEE;

  GflashShowino *theShowino;
  GflashHistogram *theHisto;
  std::vector<GflashHit> theGflashHitList;
};

#endif
