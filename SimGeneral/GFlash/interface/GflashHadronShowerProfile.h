#ifndef GflashHadronShowerProfile_H
#define GflashHadronShowerProfile_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimGeneral/GFlash/interface/GflashHistogram.h"
#include "SimGeneral/GFlash/interface/GflashNameSpace.h"
#include "SimGeneral/GFlash/interface/GflashShowino.h"
#include "SimGeneral/GFlash/interface/GflashTrajectory.h"

#include <vector>

class GflashHit;

class GflashHadronShowerProfile {
public:
  //-------------------------
  // Constructor, destructor
  //-------------------------
  GflashHadronShowerProfile(const edm::ParameterSet &parSet);
  virtual ~GflashHadronShowerProfile();

  void initialize(int showerType,
                  double energy,
                  double globalTime,
                  double charge,
                  Gflash3Vector &position,
                  Gflash3Vector &momentum);

  virtual void loadParameters();
  void hadronicParameterization();
  GflashShowino *getGflashShowino() { return theShowino; }
  std::vector<GflashHit> &getGflashHitList() { return theGflashHitList; };

protected:
  double longitudinalProfile();
  double hoProfile(double pathLength, double refDepth);
  void doCholeskyReduction(double **cc, double **vv, const int ndim);
  void getFluctuationVector(double *lowTriangle, double *correlationVector);
  void setEnergyScale(double einc, const Gflash3Vector &ssp);

  int getNumberOfSpots(Gflash::CalorimeterNumber kCalor);
  double medianLateralArm(double depth, Gflash::CalorimeterNumber kCalor);
  Gflash3Vector locateHitPosition(GflashTrajectoryPoint &point, double lateralArm);

  double fTanh(double einc, const double *par);
  double fLnE1(double einc, const double *par);
  double depthScale(double ssp, double ssp0, double length);
  double gammaProfile(double alpha, double beta, double depth, double lengthUnit);
  double twoGammaProfile(double *par, double depth, Gflash::CalorimeterNumber kIndex);

  //  SimActivityRegistry::G4StepSignal gflash_g4StepSignal;

protected:
  edm::ParameterSet theParSet;
  double theBField;
  bool theGflashHcalOuter;

  GflashShowino *theShowino;
  GflashHistogram *theHisto;

  double energyScale[Gflash::kNumberCalorimeter];
  double averageSpotEnergy[Gflash::kNumberCalorimeter];
  double longEcal[Gflash::NPar];
  double longHcal[Gflash::NPar];
  double lateralPar[Gflash::kNumberCalorimeter][Gflash::Nrpar];

  std::vector<GflashHit> theGflashHitList;
};

#endif
